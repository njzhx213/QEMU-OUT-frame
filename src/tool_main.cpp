#include <memory>
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

// CIRCT 方言
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"

// QEMU 代码生成
#include "QEMUCodeGen.h"
// CLK 分析结果
#include "ClkAnalysisResult.h"

// TableGen 声明 (会生成 create 函数)
#define GEN_PASS_DECL_DFFDEMO
#define GEN_PASS_DECL_CLKDEPENDENCYANALYSIS
#include "Passes.h.inc"

using namespace mlir;

// 命令行选项
static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional,
    llvm::cl::desc("<input .mlir>"),
    llvm::cl::Required);

static llvm::cl::opt<std::string> outputFilename(
    "o",
    llvm::cl::desc("Output file"),
    llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));

static llvm::cl::opt<bool> runClkAnalysis(
    "analyze-clk",
    llvm::cl::desc("Run CLK dependency analysis pass"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> runDffDemo(
    "dff-demo",
    llvm::cl::desc("Run DFF demo pass"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> genQemuTest(
    "gen-qemu-test",
    llvm::cl::desc("Generate QEMU code test (basic_timer example)"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> genQemu(
    "gen-qemu",
    llvm::cl::desc("Analyze LLHD and generate QEMU device code"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> qemuOutputDir(
    "qemu-out",
    llvm::cl::desc("Output directory for generated QEMU files"),
    llvm::cl::value_desc("directory"),
    llvm::cl::init(""));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv,
      "LLHD CLK Dependency Analyzer for QEMU Conversion\n\n"
      "Usage:\n"
      "  dff-opt <input.mlir> --analyze-clk    # Run CLK analysis\n"
      "  dff-opt <input.mlir> --dff-demo       # Run DFF demo\n"
  );

  // 创建 MLIR context 并注册方言
  MLIRContext context;
  DialectRegistry registry;
  registry.insert<
    circt::hw::HWDialect,
    circt::sv::SVDialect,
    circt::comb::CombDialect,
    circt::seq::SeqDialect,
    circt::llhd::LLHDDialect,
    mlir::cf::ControlFlowDialect
  >();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  // 解析输入文件
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(inputFilename, &context);
  if (!module) {
    llvm::WithColor::error() << "failed to parse input MLIR: " << inputFilename << "\n";
    return 1;
  }

  // 创建 PassManager
  PassManager pm(&context);
  pm.enableVerifier(true);

  // 根据命令行选项添加 pass
  if (runClkAnalysis) {
    pm.addPass(createClkDependencyAnalysis());
  }
  if (runDffDemo) {
    pm.addPass(createDffDemo());
  }

  // 如果指定了 --gen-qemu-test，生成测试代码（硬编码示例）
  if (genQemuTest) {
    llvm::outs() << "========================================\n";
    llvm::outs() << "QEMU Code Generation Test (hardcoded)\n";
    llvm::outs() << "========================================\n\n";

    qemu_codegen::QEMUDeviceGenerator gen("basic_timer");
    gen.addSimpleReg("control", 32);
    gen.addSimpleReg("status", 32);
    gen.addSimpleReg("load_value", 32);
    gen.addICountCounter("counter", 32,
                         qemu_codegen::CounterDirection::DOWN, 1);

    llvm::outs() << "/* ==================== basic_timer.h ==================== */\n\n";
    gen.generateHeader(llvm::outs());
    llvm::outs() << "\n\n";
    llvm::outs() << "/* ==================== basic_timer.c ==================== */\n\n";
    gen.generateSource(llvm::outs());

    return 0;
  }

  // 如果指定了 --gen-qemu，自动分析 LLHD 并生成 QEMU 代码
  if (genQemu) {
    llvm::outs() << "========================================\n";
    llvm::outs() << "QEMU Code Generation (from LLHD analysis)\n";
    llvm::outs() << "========================================\n\n";

    // 分析模块（包括事件处理逻辑）
    auto analysisResult = clk_analysis::analyzeModuleWithEvents(*module);

    llvm::outs() << "Analyzed module: " << analysisResult.moduleName << "\n";
    llvm::outs() << "Found " << analysisResult.signals.size() << " signals\n";
    llvm::outs() << "Found " << analysisResult.inputSignals.size() << " input signals\n";
    llvm::outs() << "Found " << analysisResult.eventHandlers.size() << " event handlers\n\n";

    // 打印分析结果
    llvm::outs() << "Signal Classifications:\n";
    for (const auto &sig : analysisResult.signals) {
      llvm::outs() << "  - " << sig.name << " (i" << sig.bitWidth << "): ";
      switch (sig.classification) {
        case clk_analysis::DrvClassification::CLK_IGNORABLE:
          llvm::outs() << "CLK_IGNORABLE -> SimpleReg\n";
          break;
        case clk_analysis::DrvClassification::CLK_ACCUMULATE:
          llvm::outs() << "CLK_ACCUMULATE -> ICountCounter (";
          llvm::outs() << (sig.direction == clk_analysis::AccumulateDirection::UP ? "+" : "-");
          llvm::outs() << sig.stepValue << ")\n";
          break;
        case clk_analysis::DrvClassification::CLK_LOOP_ITER:
          llvm::outs() << "CLK_LOOP_ITER -> skip (combinational)\n";
          break;
        case clk_analysis::DrvClassification::CLK_COMPLEX:
          llvm::outs() << "CLK_COMPLEX -> skip (needs manual handling)\n";
          break;
      }
    }

    // 打印输入信号
    if (!analysisResult.inputSignals.empty()) {
      llvm::outs() << "\nInput Signals (event triggers):\n";
      for (const auto &input : analysisResult.inputSignals) {
        llvm::outs() << "  - " << input.first << "\n";
      }
    }

    // 打印事件处理器
    if (!analysisResult.eventHandlers.empty()) {
      llvm::outs() << "\nEvent Handlers:\n";
      for (const auto &handler : analysisResult.eventHandlers) {
        llvm::outs() << "  - on_" << handler.triggerSignal << "_write: "
                     << handler.branches.size() << " branches\n";
      }
    }
    llvm::outs() << "\n";

    // 创建 QEMU 代码生成器
    qemu_codegen::QEMUDeviceGenerator gen(analysisResult.moduleName);

    // 根据分析结果添加信号
    for (const auto &sig : analysisResult.signals) {
      switch (sig.classification) {
        case clk_analysis::DrvClassification::CLK_IGNORABLE:
          gen.addSimpleReg(sig.name, sig.bitWidth);
          break;
        case clk_analysis::DrvClassification::CLK_ACCUMULATE: {
          auto dir = (sig.direction == clk_analysis::AccumulateDirection::UP)
                         ? qemu_codegen::CounterDirection::UP
                         : qemu_codegen::CounterDirection::DOWN;
          gen.addICountCounter(sig.name, sig.bitWidth, dir, sig.stepValue);
          break;
        }
        case clk_analysis::DrvClassification::CLK_LOOP_ITER:
        case clk_analysis::DrvClassification::CLK_COMPLEX:
          // 跳过这些，需要手动处理
          break;
      }
    }

    // 添加输入信号
    for (const auto &input : analysisResult.inputSignals) {
      gen.addInputSignal(input.first, 32);  // 默认 32 位
    }

    // 添加事件处理器
    for (const auto &handler : analysisResult.eventHandlers) {
      gen.addEventHandler(handler);
    }

    // 添加派生信号
    for (const auto &derived : analysisResult.derivedSignals) {
      gen.addDerivedSignal(derived);
    }

    // 添加控制关系
    for (const auto &rel : analysisResult.controlRelations) {
      gen.addControlRelation(rel);
    }

    // 生成代码
    if (!qemuOutputDir.empty()) {
      // 输出到文件
      std::string headerPath = qemuOutputDir.getValue() + "/" + analysisResult.moduleName + ".h";
      std::string sourcePath = qemuOutputDir.getValue() + "/" + analysisResult.moduleName + ".c";

      std::error_code ec;
      llvm::raw_fd_ostream headerFile(headerPath, ec, llvm::sys::fs::OF_Text);
      if (ec) {
        llvm::WithColor::error() << "could not open " << headerPath << ": " << ec.message() << "\n";
        return 1;
      }
      gen.generateHeader(headerFile);
      llvm::outs() << "Generated: " << headerPath << "\n";

      llvm::raw_fd_ostream sourceFile(sourcePath, ec, llvm::sys::fs::OF_Text);
      if (ec) {
        llvm::WithColor::error() << "could not open " << sourcePath << ": " << ec.message() << "\n";
        return 1;
      }
      gen.generateSource(sourceFile);
      llvm::outs() << "Generated: " << sourcePath << "\n";
    } else {
      // 输出到终端
      llvm::outs() << "/* ==================== " << analysisResult.moduleName << ".h ==================== */\n\n";
      gen.generateHeader(llvm::outs());
      llvm::outs() << "\n\n";
      llvm::outs() << "/* ==================== " << analysisResult.moduleName << ".c ==================== */\n\n";
      gen.generateSource(llvm::outs());
    }

    return 0;
  }

  // 如果没有指定任何 pass，默认运行 CLK 分析
  if (!runClkAnalysis && !runDffDemo) {
    llvm::outs() << "No pass specified, running --analyze-clk by default.\n\n";
    pm.addPass(createClkDependencyAnalysis());
  }

  // 运行 passes
  if (failed(pm.run(*module))) {
    llvm::WithColor::error() << "pass execution failed\n";
    return 1;
  }

  // 输出结果（如果需要）
  if (outputFilename != "-") {
    std::error_code ec;
    llvm::raw_fd_ostream os(outputFilename, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::WithColor::error() << "could not open output: " << ec.message() << "\n";
      return 1;
    }
    module->print(os);
  }

  return 0;
}
