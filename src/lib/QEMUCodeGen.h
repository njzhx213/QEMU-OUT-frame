#ifndef QEMU_CODEGEN_H
#define QEMU_CODEGEN_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "ClkAnalysisResult.h"
#include <string>
#include <vector>

namespace qemu_codegen {

//===----------------------------------------------------------------------===//
// 信号转换信息
//===----------------------------------------------------------------------===//

/// 信号的 QEMU 转换类型
enum class QEMUSignalType {
  SIMPLE_REG,       // 简单寄存器，直接存储
  ICOUNT_COUNTER,   // 使用 icount 的计数器
  PTIMER_COUNTER,   // 使用 ptimer 的计数器（更精确）
  MMIO_REG,         // MMIO 映射的寄存器
};

/// 计数器方向
enum class CounterDirection {
  UP,    // 递增 (counter++)
  DOWN,  // 递减 (counter--)
};

/// 单个信号的转换信息
struct SignalInfo {
  std::string name;           // 信号名
  int bitWidth;               // 位宽
  QEMUSignalType type;        // QEMU 转换类型
  CounterDirection direction; // 计数方向（仅对 counter 有效）
  int stepValue;              // 步进值（默认 1）

  SignalInfo(llvm::StringRef n, int w);
};

//===----------------------------------------------------------------------===//
// QEMU 设备代码生成器
//===----------------------------------------------------------------------===//

class QEMUDeviceGenerator {
public:
  explicit QEMUDeviceGenerator(llvm::StringRef deviceName);

  /// 添加简单寄存器
  void addSimpleReg(llvm::StringRef name, int bitWidth);

  /// 添加 icount 计数器
  void addICountCounter(llvm::StringRef name, int bitWidth,
                        CounterDirection dir, int step = 1);

  /// 添加事件处理器
  void addEventHandler(const clk_analysis::EventHandler &handler);

  /// 添加输入信号（用于事件触发）
  void addInputSignal(llvm::StringRef name, int bitWidth);

  /// 添加派生信号
  void addDerivedSignal(const clk_analysis::DerivedSignal &derived);

  /// 添加控制关系（输入信号控制计数器启停）
  void addControlRelation(const clk_analysis::ControlRelation &relation);

  /// 生成 QEMU 设备头文件
  void generateHeader(llvm::raw_ostream &os);

  /// 生成 QEMU 设备实现
  void generateSource(llvm::raw_ostream &os);

private:
  std::string deviceName_;
  std::vector<SignalInfo> signals_;
  std::vector<clk_analysis::EventHandler> eventHandlers_;
  std::vector<std::pair<std::string, int>> inputSignals_;
  std::vector<clk_analysis::DerivedSignal> derivedSignals_;
  std::vector<clk_analysis::ControlRelation> controlRelations_;

  /// 生成 ptimer 回调函数
  void generatePtimerCallback(llvm::raw_ostream &os);

  /// 生成 ptimer 计数器读取函数
  void generatePtimerRead(llvm::raw_ostream &os, const SignalInfo &sig);

  /// 生成 ptimer 计数器写入函数
  void generatePtimerWrite(llvm::raw_ostream &os, const SignalInfo &sig);

  /// 生成事件处理函数
  void generateEventHandlers(llvm::raw_ostream &os);

  /// 生成单个事件处理函数
  void generateEventHandler(llvm::raw_ostream &os,
                            const clk_analysis::EventHandler &handler);

  /// 生成条件分支代码
  /// parentCondType: 父级条件类型，用于消除冗余条件
  void generateConditionCode(llvm::raw_ostream &os,
                             const clk_analysis::ConditionalBranch &branch,
                             int indent,
                             clk_analysis::ConditionType parentCondType = clk_analysis::ConditionType::NONE);

  /// 检查信号是否是 ptimer 计数器
  bool isCounterSignal(llvm::StringRef name) const;

  /// 检查信号是否存在于状态结构体中
  bool signalExists(llvm::StringRef name) const;

  /// 检查信号是否是派生信号
  const clk_analysis::DerivedSignal* getDerivedSignal(llvm::StringRef name) const;

  /// 生成信号读取表达式（对 ptimer 计数器使用 get_ 函数）
  std::string getSignalReadExpr(llvm::StringRef name) const;

  /// 获取控制信号控制的计数器列表
  std::vector<std::string> getControlledCounters(llvm::StringRef controlSignal,
                                                  bool activeHigh) const;

  /// 生成 ptimer 启停代码
  void generatePtimerStartStop(llvm::raw_ostream &os,
                                llvm::StringRef counterName,
                                bool start, int indentLevel);

  /// 生成派生信号的 getter 函数
  void generateDerivedSignalGetters(llvm::raw_ostream &os);

  /// 生成动作代码
  void generateActionCode(llvm::raw_ostream &os,
                          const clk_analysis::EventAction &action,
                          int indent);

  /// 生成 MMIO 读函数
  void generateMMIORead(llvm::raw_ostream &os);

  /// 生成 MMIO 写函数
  void generateMMIOWrite(llvm::raw_ostream &os);

  /// 生成设备初始化代码
  void generateDeviceInit(llvm::raw_ostream &os);

  /// 获取 C 类型
  static const char* getCType(int bitWidth);

  /// 转换为大写
  static std::string toUpperCase(llvm::StringRef s);

  /// 清洗信号名（将 . 替换为 _，确保是有效的 C 标识符）
  static std::string sanitizeName(llvm::StringRef name);

  /// 生成缩进
  static void indent(llvm::raw_ostream &os, int level);
};

} // namespace qemu_codegen

#endif // QEMU_CODEGEN_H
