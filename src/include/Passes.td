include "mlir/Pass/PassBase.td"

def DffDemo : Pass<"dff-demo", "mlir::ModuleOp"> {
  let summary = "Demo pass: walk and rewrite CIRCT HW/SV ops";
  let description = [{
    A minimal pass that walks CIRCT HW/SV dialect ops and applies a tiny rewrite.
  }];
}

def ClkDependencyAnalysis : Pass<"clk-dependency-analysis", "mlir::ModuleOp"> {
  let summary = "Analyze clock dependency in LLHD IR for QEMU conversion";
  let description = [{
    This pass analyzes LLHD IR to determine which registers depend on clock
    signals and how they should be converted for QEMU device emulation.

    It classifies each register write (llhd.drv) into:
    - CLK_IGNORABLE: Can be converted to event-driven (e.g., reset, load)
    - CLK_ACCUMULATE: Needs icount/ptimer (e.g., counter++)
    - CLK_LOOP_ITER: Is a for-loop iterator, not real clock accumulation
    - CLK_COMPLEX: Complex dependency, needs further analysis
  }];
}
