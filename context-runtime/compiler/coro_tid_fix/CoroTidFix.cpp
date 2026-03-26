/*
 * CoroTidFix — LLVM pass plugin for GPU coroutine threadIdx.x preservation.
 *
 * Root cause: LLVM's @llvm.nvvm.read.ptx.sreg.tid.x intrinsic is marked
 * readnone (IntrNoMem), which allows the optimizer to CSE (common
 * subexpression eliminate) multiple calls into one. On GPU, each warp lane
 * should get a different value from this register, but CSE merges them.
 *
 * This is harmless in normal code (the optimizer sees one function on one
 * thread), but in coroutine-split code where the outlined .resume function
 * is called by 32 lanes simultaneously, the CSE'd single read is broadcast
 * to all lanes.
 *
 * Fix: Replace all @llvm.nvvm.read.ptx.sreg.tid.x calls with inline
 * assembly "mov.u32 $0, %tid.x;" marked as volatile/sideeffect, which
 * prevents CSE. Each lane then reads its own register value.
 *
 * Only applies to nvptx/nvptx64 targets.
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

/// Replace @llvm.nvvm.read.ptx.sreg.tid.x with volatile inline asm.
static bool replaceTidXWithAsm(Function &F) {
  SmallVector<CallInst *, 8> TidCalls;

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        if (auto *Callee = CI->getCalledFunction()) {
          if (Callee->getIntrinsicID() == Intrinsic::nvvm_read_ptx_sreg_tid_x) {
            TidCalls.push_back(CI);
          }
        }
      }
    }
  }

  if (TidCalls.empty()) return false;

  // Create inline asm type: () -> i32, with side effects to prevent CSE
  auto *I32Ty = Type::getInt32Ty(F.getContext());
  auto *AsmFTy = FunctionType::get(I32Ty, false);
  auto *AsmVal = InlineAsm::get(
      AsmFTy, "mov.u32 $0, %tid.x;", "=r",
      /*hasSideEffects=*/true);

  for (auto *CI : TidCalls) {
    IRBuilder<> Builder(CI);
    auto *AsmCall = Builder.CreateCall(AsmFTy, AsmVal, {}, "tid.x.asm");
    CI->replaceAllUsesWith(AsmCall);
    CI->eraseFromParent();
  }

  return true;
}

static bool isKernelFunction(const Function &F, const Module &M) {
  auto *NvvmAnnot = M.getNamedMetadata("nvvm.annotations");
  if (!NvvmAnnot) return false;
  for (auto *Op : NvvmAnnot->operands()) {
    if (Op->getNumOperands() >= 3) {
      if (auto *FnMD = dyn_cast<ValueAsMetadata>(Op->getOperand(0))) {
        if (FnMD->getValue() == &F) {
          if (auto *KindMD = dyn_cast<MDString>(Op->getOperand(1))) {
            if (KindMD->getString() == "kernel") return true;
          }
        }
      }
    }
  }
  return false;
}

struct CoroTidFixPass : PassInfoMixin<CoroTidFixPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    StringRef Triple = M.getTargetTriple();
    if (!Triple.starts_with("nvptx"))
      return PreservedAnalyses::all();

    bool Changed = false;
    unsigned Fixed = 0;

    for (auto &F : M) {
      if (F.isDeclaration()) continue;
      if (replaceTidXWithAsm(F)) {
        Changed = true;
        Fixed++;
      }
    }

    errs() << "[CoroTidFix] " << M.getName()
           << ": replaced tid.x in " << Fixed << " functions\n";

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

} // anonymous namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "CoroTidFix", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            // Run at multiple points to catch tid.x reads at all stages
            PB.registerPipelineStartEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel) {
                  MPM.addPass(CoroTidFixPass());
                });
            PB.registerOptimizerLastEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel) {
                  MPM.addPass(CoroTidFixPass());
                });
          }};
}
