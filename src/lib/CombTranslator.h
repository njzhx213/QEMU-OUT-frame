#ifndef COMB_TRANSLATOR_H
#define COMB_TRANSLATOR_H

#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include <string>
#include <optional>

namespace comb_translator {

//===----------------------------------------------------------------------===//
// Comb Dialect to C Expression Translator
//
// 支持的操作：
//   算术: add, sub, mul, divs, divu, mods, modu
//   位运算: and, or, xor, shl, shrs, shru
//   比较: icmp (eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge)
//   数据操作: extract, concat, replicate, mux, reverse, parity
//
// 参考: https://circt.llvm.org/docs/Dialects/Comb/
//===----------------------------------------------------------------------===//

/// 翻译结果
struct TranslateResult {
  bool success;           // 是否成功翻译
  std::string expr;       // C 表达式字符串
  std::string errorMsg;   // 失败时的错误信息

  static TranslateResult ok(const std::string &e) {
    return {true, e, ""};
  }
  static TranslateResult fail(const std::string &msg) {
    return {false, "", msg};
  }
};

//===----------------------------------------------------------------------===//
// 辅助函数
//===----------------------------------------------------------------------===//

/// 从 SSA 值获取名称（用于没有 name 属性的信号）
/// 使用简单的打印方法获取 SSA 名称
inline std::string getSSANameFromValue(mlir::Value val) {
  // 对于 OpResult，获取定义操作的结果名
  if (auto opResult = mlir::dyn_cast<mlir::OpResult>(val)) {
    mlir::Operation *defOp = opResult.getOwner();
    // 尝试获取 sym_name 或其他命名属性
    if (auto nameAttr = defOp->getAttrOfType<mlir::StringAttr>("name")) {
      return nameAttr.getValue().str();
    }
    if (auto symName = defOp->getAttrOfType<mlir::StringAttr>("sym_name")) {
      return symName.getValue().str();
    }
    // 使用操作的位置信息构造名称
    // 打印完整的 Value 到字符串，然后提取名称部分
    std::string str;
    llvm::raw_string_ostream os(str);
    val.print(os);
    os.flush();
    // 格式通常是 "%name" 或 "<block argument>"
    if (!str.empty() && str[0] == '%') {
      // 提取 %name 部分（到空格或冒号为止）
      size_t end = str.find_first_of(" :");
      if (end != std::string::npos) {
        return str.substr(1, end - 1);
      }
      return str.substr(1);
    }
  }
  return "";
}

/// 检测位提取模式的结果
struct BitExtractPattern {
  bool isPattern;           // 是否匹配位提取模式
  std::string signalName;   // 被提取的信号名
  bool usesBlockArgument;   // 索引是否是 BlockArgument

  static BitExtractPattern none() {
    return {false, "", false};
  }
  static BitExtractPattern match(const std::string &sig, bool usesArg) {
    return {true, sig, usesArg};
  }
};

/// 检测 (signal >> index) & 1 模式
/// 如果索引是 BlockArgument，则标记 usesBlockArgument = true
inline BitExtractPattern detectBitExtractPattern(mlir::Value val) {
  // 模式1: comb.extract (signal >> blockArg) from 0 : (i32) -> i1
  if (auto extractOp = val.getDefiningOp<circt::comb::ExtractOp>()) {
    if (extractOp.getLowBit() == 0 &&
        extractOp.getType().getIntOrFloatBitWidth() == 1) {
      mlir::Value input = extractOp.getInput();

      // 检查是否是 shru 操作
      if (auto shruOp = input.getDefiningOp<circt::comb::ShrUOp>()) {
        mlir::Value signal = shruOp.getLhs();
        mlir::Value index = shruOp.getRhs();

        // 检查信号是否来自 llhd.prb
        if (auto prbOp = signal.getDefiningOp<circt::llhd::PrbOp>()) {
          mlir::Value sig = prbOp.getSignal();
          std::string sigName;
          if (auto sigOp = sig.getDefiningOp()) {
            if (auto nameAttr = sigOp->getAttrOfType<mlir::StringAttr>("name")) {
              sigName = nameAttr.getValue().str();
            } else {
              // 使用 SSA 名称作为后备
              sigName = getSSANameFromValue(sig);
            }
          }

          // 检查索引是否是 BlockArgument
          bool usesBlockArg = mlir::isa<mlir::BlockArgument>(index);

          // 也可能索引是通过 prb 读取的循环迭代器信号
          if (!usesBlockArg) {
            if (auto idxPrb = index.getDefiningOp<circt::llhd::PrbOp>()) {
              mlir::Value idxSig = idxPrb.getSignal();
              if (auto idxSigOp = idxSig.getDefiningOp()) {
                std::string idxName;
                if (auto nameAttr = idxSigOp->getAttrOfType<mlir::StringAttr>("name")) {
                  idxName = nameAttr.getValue().str();
                } else {
                  idxName = getSSANameFromValue(idxSig);
                }
                // 如果是 int_k 这样的循环迭代器
                if (idxName.find("int_k") != std::string::npos ||
                    idxName.find("_k") != std::string::npos ||
                    idxName.find("_i") != std::string::npos) {
                  usesBlockArg = true;  // 视为循环迭代器
                }
              }
            }
          }

          if (!sigName.empty()) {
            return BitExtractPattern::match(sigName, usesBlockArg);
          }
        }
      }
    }
  }
  return BitExtractPattern::none();
}

/// 获取信号名称（穿透 llhd.prb）
inline std::string getSignalName(mlir::Value val) {
  // 直接是 llhd.prb
  if (auto prbOp = val.getDefiningOp<circt::llhd::PrbOp>()) {
    mlir::Value sig = prbOp.getSignal();
    if (auto sigOp = sig.getDefiningOp()) {
      if (auto nameAttr = sigOp->getAttrOfType<mlir::StringAttr>("name")) {
        return "s->" + nameAttr.getValue().str();
      }
      // 使用 SSA 名称作为后备
      std::string ssaName = getSSANameFromValue(sig);
      if (!ssaName.empty()) {
        return "s->" + ssaName;
      }
    }
    return "s->unnamed";
  }

  // 常量
  if (auto constOp = val.getDefiningOp<circt::hw::ConstantOp>()) {
    llvm::APInt v = constOp.getValue();
    if (v.getBitWidth() <= 64) {
      return std::to_string(v.getZExtValue());
    }
    // 大常量用十六进制
    llvm::SmallString<64> str;
    v.toStringUnsigned(str, 16);
    return "0x" + str.str().str();
  }

  // BlockArgument（函数参数）
  if (mlir::isa<mlir::BlockArgument>(val)) {
    return "arg" + std::to_string(mlir::cast<mlir::BlockArgument>(val).getArgNumber());
  }

  return "";  // 无法直接获取名称
}

/// 生成位掩码
inline std::string genMask(unsigned width) {
  if (width >= 64) {
    return "0xFFFFFFFFFFFFFFFFULL";
  }
  uint64_t mask = (1ULL << width) - 1;
  if (mask <= 0xFFFF) {
    return "0x" + llvm::utohexstr(mask);
  }
  return "0x" + llvm::utohexstr(mask) + "ULL";
}

//===----------------------------------------------------------------------===//
// 表达式翻译器（递归）
//===----------------------------------------------------------------------===//

/// 递归翻译 Value 到 C 表达式
/// maxDepth 限制递归深度，防止无限递归
inline TranslateResult translateValue(mlir::Value val, unsigned maxDepth = 10) {
  if (maxDepth == 0) {
    return TranslateResult::fail("max recursion depth exceeded");
  }

  // 1. 直接获取名称（信号或常量）
  std::string name = getSignalName(val);
  if (!name.empty()) {
    return TranslateResult::ok(name);
  }

  // 2. 获取定义操作
  mlir::Operation *defOp = val.getDefiningOp();
  if (!defOp) {
    return TranslateResult::fail("no defining op");
  }

  // ========== 算术操作 ==========

  // comb.add: 加法（variadic）
  if (auto addOp = mlir::dyn_cast<circt::comb::AddOp>(defOp)) {
    std::string result;
    for (unsigned i = 0; i < addOp.getNumOperands(); ++i) {
      auto sub = translateValue(addOp.getOperand(i), maxDepth - 1);
      if (!sub.success) return sub;
      if (i > 0) result += " + ";
      result += "(" + sub.expr + ")";
    }
    return TranslateResult::ok(result);
  }

  // comb.sub: 减法
  if (auto subOp = mlir::dyn_cast<circt::comb::SubOp>(defOp)) {
    auto lhs = translateValue(subOp.getLhs(), maxDepth - 1);
    auto rhs = translateValue(subOp.getRhs(), maxDepth - 1);
    if (!lhs.success) return lhs;
    if (!rhs.success) return rhs;
    return TranslateResult::ok("(" + lhs.expr + ") - (" + rhs.expr + ")");
  }

  // comb.mul: 乘法（variadic）
  if (auto mulOp = mlir::dyn_cast<circt::comb::MulOp>(defOp)) {
    std::string result;
    for (unsigned i = 0; i < mulOp.getNumOperands(); ++i) {
      auto sub = translateValue(mulOp.getOperand(i), maxDepth - 1);
      if (!sub.success) return sub;
      if (i > 0) result += " * ";
      result += "(" + sub.expr + ")";
    }
    return TranslateResult::ok(result);
  }

  // comb.divu: 无符号除法
  if (auto divuOp = mlir::dyn_cast<circt::comb::DivUOp>(defOp)) {
    auto lhs = translateValue(divuOp.getLhs(), maxDepth - 1);
    auto rhs = translateValue(divuOp.getRhs(), maxDepth - 1);
    if (!lhs.success) return lhs;
    if (!rhs.success) return rhs;
    return TranslateResult::ok("(" + lhs.expr + ") / (" + rhs.expr + ")");
  }

  // comb.divs: 有符号除法
  if (auto divsOp = mlir::dyn_cast<circt::comb::DivSOp>(defOp)) {
    auto lhs = translateValue(divsOp.getLhs(), maxDepth - 1);
    auto rhs = translateValue(divsOp.getRhs(), maxDepth - 1);
    if (!lhs.success) return lhs;
    if (!rhs.success) return rhs;
    return TranslateResult::ok("((int64_t)(" + lhs.expr + ")) / ((int64_t)(" + rhs.expr + "))");
  }

  // comb.modu: 无符号取模
  if (auto moduOp = mlir::dyn_cast<circt::comb::ModUOp>(defOp)) {
    auto lhs = translateValue(moduOp.getLhs(), maxDepth - 1);
    auto rhs = translateValue(moduOp.getRhs(), maxDepth - 1);
    if (!lhs.success) return lhs;
    if (!rhs.success) return rhs;
    return TranslateResult::ok("(" + lhs.expr + ") % (" + rhs.expr + ")");
  }

  // comb.mods: 有符号取模
  if (auto modsOp = mlir::dyn_cast<circt::comb::ModSOp>(defOp)) {
    auto lhs = translateValue(modsOp.getLhs(), maxDepth - 1);
    auto rhs = translateValue(modsOp.getRhs(), maxDepth - 1);
    if (!lhs.success) return lhs;
    if (!rhs.success) return rhs;
    return TranslateResult::ok("((int64_t)(" + lhs.expr + ")) % ((int64_t)(" + rhs.expr + "))");
  }

  // ========== 位运算 ==========

  // comb.and: 按位与（variadic）
  if (auto andOp = mlir::dyn_cast<circt::comb::AndOp>(defOp)) {
    std::string result;
    for (unsigned i = 0; i < andOp.getNumOperands(); ++i) {
      auto sub = translateValue(andOp.getOperand(i), maxDepth - 1);
      if (!sub.success) return sub;
      if (i > 0) result += " & ";
      result += "(" + sub.expr + ")";
    }
    return TranslateResult::ok(result);
  }

  // comb.or: 按位或（variadic）
  if (auto orOp = mlir::dyn_cast<circt::comb::OrOp>(defOp)) {
    std::string result;
    for (unsigned i = 0; i < orOp.getNumOperands(); ++i) {
      auto sub = translateValue(orOp.getOperand(i), maxDepth - 1);
      if (!sub.success) return sub;
      if (i > 0) result += " | ";
      result += "(" + sub.expr + ")";
    }
    return TranslateResult::ok(result);
  }

  // comb.xor: 按位异或（variadic）
  if (auto xorOp = mlir::dyn_cast<circt::comb::XorOp>(defOp)) {
    // 特殊情况：XOR 1 = 取反
    if (xorOp.getNumOperands() == 2) {
      if (auto constOp = xorOp.getOperand(1).getDefiningOp<circt::hw::ConstantOp>()) {
        if (constOp.getValue().isAllOnes()) {
          auto inner = translateValue(xorOp.getOperand(0), maxDepth - 1);
          if (!inner.success) return inner;
          return TranslateResult::ok("~(" + inner.expr + ")");
        }
      }
    }
    std::string result;
    for (unsigned i = 0; i < xorOp.getNumOperands(); ++i) {
      auto sub = translateValue(xorOp.getOperand(i), maxDepth - 1);
      if (!sub.success) return sub;
      if (i > 0) result += " ^ ";
      result += "(" + sub.expr + ")";
    }
    return TranslateResult::ok(result);
  }

  // comb.shl: 左移
  if (auto shlOp = mlir::dyn_cast<circt::comb::ShlOp>(defOp)) {
    auto lhs = translateValue(shlOp.getLhs(), maxDepth - 1);
    auto rhs = translateValue(shlOp.getRhs(), maxDepth - 1);
    if (!lhs.success) return lhs;
    if (!rhs.success) return rhs;
    return TranslateResult::ok("(" + lhs.expr + ") << (" + rhs.expr + ")");
  }

  // comb.shru: 无符号右移
  if (auto shruOp = mlir::dyn_cast<circt::comb::ShrUOp>(defOp)) {
    auto lhs = translateValue(shruOp.getLhs(), maxDepth - 1);
    auto rhs = translateValue(shruOp.getRhs(), maxDepth - 1);
    if (!lhs.success) return lhs;
    if (!rhs.success) return rhs;
    return TranslateResult::ok("(" + lhs.expr + ") >> (" + rhs.expr + ")");
  }

  // comb.shrs: 有符号右移
  if (auto shrsOp = mlir::dyn_cast<circt::comb::ShrSOp>(defOp)) {
    auto lhs = translateValue(shrsOp.getLhs(), maxDepth - 1);
    auto rhs = translateValue(shrsOp.getRhs(), maxDepth - 1);
    if (!lhs.success) return lhs;
    if (!rhs.success) return rhs;
    return TranslateResult::ok("((int64_t)(" + lhs.expr + ")) >> (" + rhs.expr + ")");
  }

  // ========== 数据操作 ==========

  // comb.extract: 位提取
  // extract %input from lowBit : (iN) -> iM
  // C: (input >> lowBit) & mask
  if (auto extractOp = mlir::dyn_cast<circt::comb::ExtractOp>(defOp)) {
    auto input = translateValue(extractOp.getInput(), maxDepth - 1);
    if (!input.success) return input;

    unsigned lowBit = extractOp.getLowBit();
    unsigned width = extractOp.getType().getIntOrFloatBitWidth();

    if (lowBit == 0 && width == 1) {
      // 特殊情况：extract from 0 : i1 → & 1
      return TranslateResult::ok("((" + input.expr + ") & 1)");
    } else if (lowBit == 0) {
      // 特殊情况：从 0 开始提取
      return TranslateResult::ok("((" + input.expr + ") & " + genMask(width) + ")");
    } else {
      return TranslateResult::ok("(((" + input.expr + ") >> " +
                                  std::to_string(lowBit) + ") & " + genMask(width) + ")");
    }
  }

  // comb.concat: 位拼接
  // concat %hi, %lo : (iN, iM) -> i(N+M)
  // C: (hi << M) | lo
  if (auto concatOp = mlir::dyn_cast<circt::comb::ConcatOp>(defOp)) {
    if (concatOp.getNumOperands() == 0) {
      return TranslateResult::fail("empty concat");
    }

    // 从高位到低位处理
    std::string result;
    unsigned accumulatedWidth = 0;

    // concat 的操作数顺序：第一个是最高位
    for (int i = concatOp.getNumOperands() - 1; i >= 0; --i) {
      mlir::Value operand = concatOp.getOperand(i);
      auto sub = translateValue(operand, maxDepth - 1);
      if (!sub.success) return sub;

      unsigned opWidth = operand.getType().getIntOrFloatBitWidth();

      if (result.empty()) {
        result = "(" + sub.expr + ")";
      } else {
        result = "((" + sub.expr + ") << " + std::to_string(accumulatedWidth) + ") | " + result;
      }
      accumulatedWidth += opWidth;
    }
    return TranslateResult::ok("(" + result + ")");
  }

  // comb.replicate: 位复制
  // replicate %bit : (i1) -> i32
  // C: bit ? 0xFFFFFFFF : 0
  if (auto repOp = mlir::dyn_cast<circt::comb::ReplicateOp>(defOp)) {
    auto input = translateValue(repOp.getInput(), maxDepth - 1);
    if (!input.success) return input;

    unsigned resultWidth = repOp.getType().getIntOrFloatBitWidth();
    std::string allOnes = genMask(resultWidth);

    return TranslateResult::ok("((" + input.expr + ") ? " + allOnes + " : 0)");
  }

  // comb.mux: 多路选择器
  // mux %cond, %true, %false : iN
  // C: cond ? trueVal : falseVal
  if (auto muxOp = mlir::dyn_cast<circt::comb::MuxOp>(defOp)) {
    // 优化：检测 mux(cond, a[i], b[i]) 模式，简化为 cond ? a : b
    auto truePattern = detectBitExtractPattern(muxOp.getTrueValue());
    auto falsePattern = detectBitExtractPattern(muxOp.getFalseValue());

    if (truePattern.isPattern && falsePattern.isPattern &&
        truePattern.usesBlockArgument && falsePattern.usesBlockArgument) {
      // 两边都是用 BlockArgument 索引的位提取，简化为整数级别操作
      auto cond = translateValue(muxOp.getCond(), maxDepth - 1);
      if (!cond.success) return cond;
      return TranslateResult::ok("((" + cond.expr + ") ? (s->" +
                                  truePattern.signalName + ") : (s->" +
                                  falsePattern.signalName + "))");
    }

    auto cond = translateValue(muxOp.getCond(), maxDepth - 1);
    auto trueVal = translateValue(muxOp.getTrueValue(), maxDepth - 1);
    auto falseVal = translateValue(muxOp.getFalseValue(), maxDepth - 1);
    if (!cond.success) return cond;
    if (!trueVal.success) return trueVal;
    if (!falseVal.success) return falseVal;

    return TranslateResult::ok("((" + cond.expr + ") ? (" + trueVal.expr + ") : (" + falseVal.expr + "))");
  }

  // comb.reverse: 位反转
  if (auto revOp = mlir::dyn_cast<circt::comb::ReverseOp>(defOp)) {
    auto input = translateValue(revOp.getInput(), maxDepth - 1);
    if (!input.success) return input;
    unsigned width = revOp.getType().getIntOrFloatBitWidth();
    // 生成位反转辅助函数调用
    return TranslateResult::ok("bit_reverse" + std::to_string(width) + "(" + input.expr + ")");
  }

  // comb.parity: 奇偶校验
  if (auto parOp = mlir::dyn_cast<circt::comb::ParityOp>(defOp)) {
    auto input = translateValue(parOp.getInput(), maxDepth - 1);
    if (!input.success) return input;
    // 使用 GCC 内置函数或手写
    return TranslateResult::ok("(__builtin_parityll(" + input.expr + "))");
  }

  // ========== 比较操作 ==========

  // comb.icmp: 整数比较
  if (auto icmpOp = mlir::dyn_cast<circt::comb::ICmpOp>(defOp)) {
    auto lhs = translateValue(icmpOp.getLhs(), maxDepth - 1);
    auto rhs = translateValue(icmpOp.getRhs(), maxDepth - 1);
    if (!lhs.success) return lhs;
    if (!rhs.success) return rhs;

    std::string op;
    bool isSigned = false;
    switch (icmpOp.getPredicate()) {
      case circt::comb::ICmpPredicate::eq:  op = "=="; break;
      case circt::comb::ICmpPredicate::ne:  op = "!="; break;
      case circt::comb::ICmpPredicate::slt: op = "<";  isSigned = true; break;
      case circt::comb::ICmpPredicate::sle: op = "<="; isSigned = true; break;
      case circt::comb::ICmpPredicate::sgt: op = ">";  isSigned = true; break;
      case circt::comb::ICmpPredicate::sge: op = ">="; isSigned = true; break;
      case circt::comb::ICmpPredicate::ult: op = "<";  break;
      case circt::comb::ICmpPredicate::ule: op = "<="; break;
      case circt::comb::ICmpPredicate::ugt: op = ">";  break;
      case circt::comb::ICmpPredicate::uge: op = ">="; break;
      default:
        return TranslateResult::fail("unknown icmp predicate");
    }

    if (isSigned) {
      return TranslateResult::ok("(((int64_t)(" + lhs.expr + ")) " + op +
                                  " ((int64_t)(" + rhs.expr + ")))");
    }
    return TranslateResult::ok("((" + lhs.expr + ") " + op + " (" + rhs.expr + "))");
  }

  // 未支持的操作
  return TranslateResult::fail("unsupported op: " + defOp->getName().getStringRef().str());
}

/// 翻译 llhd.drv 操作的值到 C 表达式
inline TranslateResult translateDrvValue(circt::llhd::DrvOp drv) {
  return translateValue(drv.getValue());
}

} // namespace comb_translator

#endif // COMB_TRANSLATOR_H
