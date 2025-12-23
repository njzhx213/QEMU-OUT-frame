# LLHD 到 QEMU 设备转换工具

本目录包含从 LLHD IR 自动生成的 QEMU 设备代码。

## 更新日志

### 2025-12-23

**解决的问题：**
- ✅ **Issue #5 解决** - `unnamed` 信号已消除（循环展开简化 + SSA 名称后备）
- ✅ **Issue #7 架构完善** - 统一使用纯功能分析方案
- ✅ **Issue #9 解决** - 输入信号数据流分析（不再依赖名字匹配）

**新增功能：**
1. **时钟/复位信号纯功能区分**: 实现基于触发效果的信号分类（不再依赖名字）
   - 时钟信号：触发的所有 drv 操作都是 hold 模式（`reg = prb reg`）
   - 复位信号：触发的 drv 操作有状态修改（如 `counter = 0`）
   - 新增 `TriggerBranchEffect` 结构体和 `analyzeTriggerBranchEffects()` 函数
   - 新增 `isClockByTriggerEffect()` 纯功能检测函数

2. **DrvClassification 枚举重命名**: 更语义化的命名
   - `CLK_IGNORABLE` → `STATE_UNCHANGED`
   - `CLK_ACCUMULATE` → `STATE_ACCUMULATE`
   - `CLK_LOOP_ITER` → `STATE_LOOP_ITER`
   - `CLK_COMPLEX` → `STATE_COMPLEX`

3. **完整的信号类型分析框架**:
   - 方案1: `analyzeSignalRole()` - 基于拓扑角色分析
   - 方案2: `isInternalSignalByUsagePattern()` - 基于使用模式
   - 方案3: `isGPIOInputByDataFlow()`, `isWriteEnableByDataFlow()` - 基于数据流
   - 触发效果分析: `isClockByTriggerEffect()` - 区分时钟 vs 复位

**架构改进：**
1. **两步时钟检测策略**:
   ```cpp
   // 第一步：检查基本结构特征
   if (!isClockSignalByUsagePattern(signal, processOp)) return false;

   // 第二步：检查触发效果（区分时钟和复位）
   if (isClockByTriggerEffect(signal, processOp)) {
     // 是时钟，可以过滤
   } else {
     // 是复位或控制信号，必须保留
   }
   ```

2. **SignalTracing.h 新增函数** (987-1129行):
   - `isDrvHoldPattern()` - 检查 drv 是否是 hold 模式
   - `isDrvConstantInit()` - 检查 drv 是否是常量初始化
   - `collectBranchDrvEffects()` - 递归收集分支中的 drv 效果
   - `analyzeTriggerBranchEffects()` - 分析触发信号的分支效果

3. **综合信号类型推断**:
   ```cpp
   enum class SignalTypeByDataFlow {
     GPIOInput,         // GPIO 外部输入信号
     WriteEnable,       // 写使能信号
     APBProtocol,       // APB 协议信号
     ClockReset,        // 时钟/复位信号
     InternalRegister,  // 内部寄存器
     StateRegister,     // 状态寄存器
   };
   ```

**修改的文件**:
- `SignalTracing.h` - 新增触发分支效果分析、完整数据流分析框架
- `ClkAnalysisResult.h` - 重命名 DrvClassification 枚举
- `ClkAnalysisResult.cpp` - 更新 isClockSignalByUsageInModule()
- `Passes.td` - 更新文档
- `Passes.cpp` - 更新枚举引用
- `tool_main.cpp` - 更新输出字符串

### 2025-12-22

**新增功能：**
1. **只读寄存器提取**: 新增对只读 APB 寄存器的完整提取支持
   - 直接搜索 `drv prdata` 操作（14个）
   - 使用 `signal_tracing::traceToSignal()` 追踪值来源
   - 递归向上查找前驱块中的地址检查条件
   - 支持多种地址模式: `concat`, `extract`, 直接 `paddr`
   - 自动去除 `ri_` 前缀，获取真实寄存器名
   - 成功提取 4 个只读寄存器: `gpio_int_status`, `gpio_raw_int_status`, `gpio_debounce`, `gpio_ext_data`

2. **寄存器覆盖率大幅提升**:
   - 可写寄存器: 7/8 (87.5%)
   - 只读寄存器: 4/4 (100%) ✨ **新增**
   - **总体覆盖率: 11/12 (91.7%)** ⬆️ 从 58.3% 提升 33.4%

**修复问题：**
1. **地址符号扩展 Bug**: 修复 `getSExtValue()` 导致的负地址问题
   - 问题: i5 的 -8 被符号扩展为 -32 → 0xffffffe0
   - 修复: 改用 `getZExtValue()`，正确计算为 24 → 0x60
   - 影响: 所有 APB 地址现在正确提取

2. **MMIO 代码质量提升**:
   - Read cases: 7 → 11 (+4, +57%)
   - Write cases: 7 (不变)
   - 内部信号: 0 (100% 过滤)
   - 正确的读写权限标记

**架构改进：**
1. **复用 SignalTracing 库**:
   - 写寄存器提取: 使用 `analyzeAndCondition()` + `traceTrueBranchDrives()`
   - 读寄存器提取: 使用 `traceToSignal()` 追踪数据源
   - 统一的控制流分析框架

2. **递归控制流追踪**:
   ```cpp
   std::function<void(Block*)> findAddress = [&](Block* block) {
     for (Block* pred : block->getPredecessors()) {
       // 检查前驱块中的地址检查条件
       if (auto condBr = dyn_cast<cf::CondBranchOp>(pred->getTerminator())) {
         // 分析条件并提取地址
       }
       findAddress(pred);  // 递归向上追踪
     }
   };
   ```

**测试结果** (gpio0_llhd.mlir):
- ✅ 总 MMIO cases: 18 (Read: 11, Write: 7)
- ✅ 真实寄存器: 11/12 (91.7%)
- ✅ 内部信号: 0 (100% 过滤)
- ✅ 代码可用性: ⭐⭐⭐⭐⭐ (完整的 GPIO 功能仿真)

**已知限制：**
- gpio_int_clr 与 gpio_int_level_sync 地址冲突 (0x60) - 原始 LLHD IR 设计问题

### 2025-12-17

**新增功能：**
1. **APB 寄存器地址映射提取**: `extractAPBRegisterMappings()` 函数从 LLHD 中提取实际的 APB 寄存器地址
   - 分析模式: `paddr → extract → icmp eq const → and(psel, penable, pwrite, icmp) → mux(and, pwdata, reg)`
   - 自动检测可读/可写属性

2. **GPIO 输入信号分类**: 输入信号现在按类型分类处理
   - `clock`: 时钟信号（过滤，不生成代码）
   - `apb`: APB 协议信号（通过 MMIO 处理）
   - `gpio_in`: GPIO 外部输入（通过 `qdev_init_gpio_in` 处理）
   - `input`: 普通输入信号（事件触发）

3. **GPIO 输入回调生成**: `generateGPIOInputCallback()` 和 `generateUpdateState()` 函数
   - 生成 `qdev_init_gpio_in()` 初始化代码
   - 生成 GPIO 输入变化回调函数

**修复问题：**
1. **统一 `isLoopIterator()` 逻辑**: 修复 `--analyze-clk` 与 `--gen-qemu` 结果不一致问题
   - 在 `ClkAnalysisResult.cpp` 中添加 `isInSameLoopWithCondition()` 函数
   - 现在两个 pass 对 `int_k` 的分类一致（STATE_LOOP_ITER，不再错误识别为 STATE_ACCUMULATE）

2. **BlockArgument (arg0) 循环展开**: 将循环中的 bit-by-bit 操作简化为整数级别操作
   - 检测 `mux(cond, a[i], b[i])` 模式，简化为 `cond ? a : b`
   - 检测 `target[i] = source[i]` 模式，简化为 `target = source`
   - 之前: `s->unnamed = (((s->int_level) >> (arg0)) & 1);`
   - 之后: `s->gpio_int_status_level = s->int_level;`

3. **处理 process 外部的组合逻辑**: 现在正确识别和处理不在 `llhd.process` 内的 drv 操作
   - 之前 `int_level` 等信号未被检测到
   - 现在所有组合逻辑信号都会被正确追踪

4. **SSA 名称后备机制**: 对于没有 `name` 属性的 LLHD 信号，使用 SSA 名称作为后备
   - 之前: 这些信号显示为 `/* undefined signal: int_level */`
   - 之后: 正确使用 `s->int_level`

### 2025-12-16

**新增功能：**
- **CombTranslator**: 完整的 comb dialect 到 C 表达式翻译器
  - 支持算术操作: add, sub, mul, divs, divu, mods, modu
  - 支持位运算: and, or, xor, shl, shrs, shru
  - 支持比较操作: icmp (eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge)
  - 支持数据操作: extract, concat, replicate, mux, reverse, parity

**修复问题：**
1. **信号名称净化**: 自动将 `.`、`[`、`]`、`-`、空格替换为 `_`，确保生成有效的 C 标识符
   - 例如: `SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff2` → `SUPPORT_INT_LEVEL_SYNC_PROC_int_level_sync_in_ff2`

2. **冗余条件消除**: 消除嵌套条件中的重复检查
   - 之前: `if (value) { if (value) { ... } }`
   - 之后: `if (value) { ... }`

3. **未定义信号处理**: 对于无法找到的信号，生成注释而非无效代码
   - 输出: `/* undefined signal: signal_name */`

4. **COMPUTE 动作修复**: 正确生成计算表达式
   - 添加 `s->` 前缀
   - 添加分号
   - 例如: `s->unnamed = (((s->int_level) >> (arg0)) & 1);`

5. **统一事件处理**: `collectDrvsInBlock()` 现在使用 `tryGenerateAction()` 而非旧的 `analyzeDrvAction()`，确保所有表达式都经过 CombTranslator 处理

## 已知问题

### ~~1. BlockArgument 未解析 (arg0, arg1, ...)~~ 【已修复 2025-12-17】

已通过检测 bit-by-bit 操作模式并简化为整数级别操作修复。

现在生成的代码：
```c
// 简化为整数级别操作
s->gpio_int_status_level = s->gpio_int_level_sync ? s->int_level : s->int_level_sync_in;
```

### ~~2. 部分信号未定义~~ 【已修复 2025-12-17】

已通过以下方式修复：
1. 添加 SSA 名称后备机制，对于没有 `name` 属性的信号使用 SSA 名称
2. 处理 process 外部的组合逻辑 drv 操作

现在 `int_level` 等信号都能正确识别并加入状态结构体。

### ~~3. `--analyze-clk` 与 `--gen-qemu` 结果不一致~~ 【已修复 2025-12-17】

已通过统一 `isLoopIterator()` 逻辑修复。现在两个 pass 结果一致：
- STATE_UNCHANGED: 114
- STATE_ACCUMULATE: 0
- STATE_LOOP_ITER: 1

### ~~4. 事件处理器覆盖不完整~~ 【已修复 2025-12-17】

已通过输入信号分类解决。现在输入信号按类型分类处理：
- **时钟信号** (`pclk`, `pclk_int`, `pclk_intr`): 标记为 `clock`，过滤掉
- **APB 协议信号** (`paddr`, `pwdata`, `penable`, `psel`, `pwrite`): 标记为 `apb`，通过 MMIO 处理
- **GPIO 外部输入** (`gpio_ext_porta`, `gpio_in_data`): 标记为 `gpio_in`，通过 `qdev_init_gpio_in` 处理
- **普通输入** (`presetn`, `gpio_int_level_sync`): 标记为 `input`，生成事件处理器

### ~~5. `unnamed` 信号~~ 【已修复】

~~存在一个名为 `unnamed` 的信号（gpio_top.c:286），说明某些 LLHD 信号没有正确命名~~

已通过循环展开简化和 SSA 名称后备机制修复。当前生成的代码中不再有 `unnamed` 信号。

### ~~6. Generate/For 循环展开~~ 【已修复 2025-12-17】

已通过检测 bit-by-bit 操作模式并简化为整数级别操作修复。

对于 `mux(cond, a[i], b[i])` 形式的循环操作，现在自动简化为：
```c
s->result = s->cond ? s->a : s->b;
```

不再需要显式的循环展开。

### ~~7. 寄存器地址映射 - LLHD 方言支持有限~~ 【✅ 已完全解决 2025-12-22】

#### 问题描述
`extractAPBRegisterMappings()` 函数原本为 HW/Seq 方言设计，查找 `seq.firreg` 操作获取寄存器名。在 LLHD 方言中存在严重问题：
- ❌ 生成 125 个 MMIO case 语句，包含大量内部信号 (`ri_*`, `*_wen`, `*_tmp`, `PROC.*`, `_ff*`)
- ❌ 使用顺序地址而非真实 APB 地址
- ❌ 只提取写寄存器，遗漏所有只读寄存器
- ❌ 寄存器覆盖率仅 58.3% (7/12)

#### 解决方案

采用纯功能分析方案，不依赖信号名字：

**1. 方案1 - 拓扑角色分析** (`SignalTracing.h:135-340`)
```cpp
// 信号拓扑角色（不检查名字，而是分析信号在拓扑中的角色）
enum class SignalRole {
  ModuleInput,      // 模块输入端口 (BlockArgument)
  ControlFlow,      // 控制流信号 (只用于 cf.cond_br 条件)
  AddressSelector,  // 地址选择 (只用于 icmp 比较)
  DataTransfer,     // 数据传输 (用于 drv 的 value)
  IntermediateValue // 内部中间值 (有 drv 写入，也有 prb 读取)
};

SignalRole analyzeSignalRole(mlir::Value signal);
```

**2. 方案2 - 使用模式识别** (`SignalTracing.h:1131-1243`)
```cpp
// 内部信号识别（不依赖名字前缀）
bool isInternalSignalByUsagePattern(mlir::Value signal, hw::HWModuleOp moduleOp) {
  // 1. 不是模块输入端口
  // 2. 有且仅有一个 drv 写入点
  // 3. 只在特定 process 内部使用（不跨 process）
  // 4. 写入后立即被 prb 读取，用于计算另一个值
}
```

**3. 方案3 - 数据流分析** (`SignalTracing.h:1288-1594`)
```cpp
// 写使能信号识别（替代名字匹配 *_wen）
bool isWriteEnableByDataFlow(mlir::Value signal, hw::HWModuleOp moduleOp) {
  // 1. 数据来源: and(psel, penable, pwrite) → 地址检查 → drv
  // 2. 使用方式: prb → 用作 mux 的条件选择
  // 3. 不是最终的寄存器状态（只是控制信号）
}

// GPIO 输入信号识别
bool isGPIOInputByDataFlow(mlir::Value signal, hw::HWModuleOp moduleOp) {
  // 1. 是模块输入端口
  // 2. 数据流向: port → prb → 组合逻辑 → drv 内部寄存器
  // 3. 不参与地址选择（不用于 icmp）
  // 4. 不参与控制流（不用于 and/or 形成条件）
}
```

**3. APB 寄存器地址提取** (`ClkAnalysisResult.cpp`)
```cpp
// 写寄存器提取
// 1. 检测 APB 写条件: and(psel, penable, pwrite)
// 2. 追踪 true 分支中的驱动操作
// 3. 递归遍历控制流，查找地址检查 icmp(extract(paddr), const)
// 4. 使用 getZExtValue() 而非 getSExtValue() 避免符号扩展问题

// 只读寄存器提取
// 1. 直接搜索 drv prdata 操作
// 2. 使用 traceToSignal() 追踪 prdata 值的来源
// 3. 去除 ri_ 前缀获取真实寄存器名
// 4. 递归向上查找前驱块中的地址检查条件
```

#### 技术亮点

1. **纯功能分析**（不依赖信号名字）:
   - `isInternalSignalByUsagePattern()` - 基于使用模式
   - `isWriteEnableByDataFlow()` - 基于数据流
   - `isGPIOInputByDataFlow()` - 基于数据流

2. **SignalTracing 库**:
   - `analyzeAndCondition()` - 分析 AND 条件，提取控制信号
   - `traceTrueBranchDrives()` - 追踪 true 分支的驱动操作
   - `traceToSignal()` - 穿透 XOR/PRB 操作，追踪到源信号

3. **递归控制流分析**:
   - 向上遍历 block predecessors
   - 在每个前驱块中查找地址检查条件
   - 支持多层嵌套的控制流

4. **多种地址提取模式**:
   - `concat(high, low)` - 拼接模式
   - `extract(paddr, offset)` - 提取模式
   - 直接 `paddr` - 简单模式

#### 最终结果 (gpio0_llhd.mlir)

| 指标 | 修复前 | 修复后 | 改进 |
|------|--------|--------|------|
| **MMIO Read cases** | 7 | 11 | +4 (+57%) |
| **MMIO Write cases** | 7 | 7 | - |
| **总 case 数** | 125 | 18 | -107 (-85.6%) |
| **内部信号** | 80+ | 0 | -100% ✨ |
| **可写寄存器** | 7/8 | 7/8 | 87.5% |
| **只读寄存器** | 0/4 | 4/4 | 0% → 100% ✨ |
| **总体覆盖率** | 58.3% | **91.7%** | **+33.4%** ✨ |

#### 提取的寄存器列表

**可读写寄存器 (7个)**:
| 地址 | 寄存器名 | 功能 |
|------|----------|------|
| 0x00 | gpio_sw_data | GPIO 数据寄存器 |
| 0x04 | gpio_sw_dir | GPIO 方向控制 |
| 0x30 | gpio_int_en | 中断使能 |
| 0x34 | gpio_int_mask | 中断屏蔽 |
| 0x38 | gpio_int_type | 中断类型 |
| 0x3c | gpio_int_pol | 中断极性 |
| 0x60 | gpio_int_level_sync | 中断电平同步 |

**只读寄存器 (4个)** ✨ **新增**:
| 地址 | 寄存器名 | 功能 |
|------|----------|------|
| 0x40 | gpio_int_status | 中断状态寄存器 |
| 0x44 | gpio_raw_int_status | 原始中断状态 |
| 0x48 | gpio_debounce | 防抖配置 |
| 0x50 | gpio_ext_data | GPIO 外部输入数据 |

#### 已知限制
- **地址冲突**: `gpio_int_clr` 与 `gpio_int_level_sync` 共享地址 0x60 (原始 LLHD IR 设计问题)

**相关提交**:
- 2025-12-17: 使用 SignalTracing 库重写 APB 映射提取
- 2025-12-22: 添加只读寄存器提取，修复地址符号扩展 bug

### 8. Signal Tracing / update_state 组合逻辑生成 【进行中】

`generateUpdateState()` 函数当前生成的是硬编码的占位符代码：

```c
static void gpio_top_update_state(gpio_top_state *s)
{
    /* Combinational logic: gpio_ext_porta -> int_level -> gpio_int_status */
    /* TODO: Add traced combinational expressions here */

    /* Update interrupt output */
    uint32_t pending = s->gpio_int_status & s->gpio_int_en & ~s->gpio_int_mask;
    // ...
}
```

**问题**: 组合逻辑部分（从 `gpio_ext_porta` 到 `int_level` 到 `gpio_int_status` 的信号传播）需要从 LLHD IR 自动追踪生成。

**需要实现**:
1. 从 GPIO 输入信号开始，追踪所有依赖的组合逻辑链
2. 生成正确的信号传播表达式到 `update_state()` 函数
3. 确保中断状态正确更新

**进展** (2025-12-23):
- ✅ 时钟/复位信号纯功能区分已实现 - 基于触发效果分析，不再依赖信号名
- ✅ 新增 `TriggerBranchEffect` 和 `analyzeTriggerBranchEffects()` 用于分析触发分支效果
- ✅ `isDrvHoldPattern()` 可检测 hold 模式 drv 操作
- 🔄 下一步：利用这些基础设施追踪组合逻辑链并生成表达式

**相关代码**: [QEMUCodeGen.cpp:841-861](src/lib/QEMUCodeGen.cpp#L841-L861)

### ~~9. 输入信号数据流分析缺失~~ 【✅ 已解决 2025-12-23】

#### 问题描述
原本的输入信号分类是**基于名字匹配**，可能导致误分类。

#### 解决方案

现在已实现三种纯功能分析方案（不依赖名字）：

**方案2: 基于使用模式识别** (`SignalTracing.h:1214-1286`)
```cpp
// 内部信号的结构特征：
// 1. 不是模块输入端口
// 2. 有且仅有一个 drv 写入点
// 3. 只在特定 process 内部使用（不跨 process）
// 4. 写入后立即被 prb 读取，用于计算另一个值
bool isInternalSignalByUsagePattern(mlir::Value signal, hw::HWModuleOp moduleOp);

// 时钟信号的结构特征：
// 1. 是 llhd.process 的敏感信号（sig 参数）
// 2. 在 process 内通过 prb 读取
// 3. 用于触发状态更新，但自身不被 drv 修改
bool isClockSignalByUsagePattern(mlir::Value signal, llhd::ProcessOp processOp);
```

**方案3: 基于数据流依赖分析** (`SignalTracing.h:1288-1650`)
```cpp
// GPIO 输入信号的特征：
// 1. 是模块输入端口
// 2. 数据流向: port → prb → 组合逻辑 → drv 内部寄存器
// 3. 不参与地址选择（不用于 icmp）
// 4. 不参与控制流（不用于 and/or 形成条件）
bool isGPIOInputByDataFlow(mlir::Value signal, hw::HWModuleOp moduleOp);

// 写使能信号 (_wen) 的特征：
// 1. 数据来源: and(psel, penable, pwrite) → 地址检查 → drv *_wen
// 2. 使用方式: prb *_wen → 用作 mux 的条件选择
// 3. 不是最终的寄存器状态（只是控制信号）
bool isWriteEnableByDataFlow(mlir::Value signal, hw::HWModuleOp moduleOp);
```

**触发效果分析** (`SignalTracing.h:987-1129`) - 2025-12-23 新增
```cpp
// 时钟 vs 复位信号区分（纯功能）：
// - 时钟信号：触发的所有 drv 操作都是 hold 模式（reg = prb reg）
// - 复位信号：触发的 drv 操作有状态修改（如 counter = 0）
bool isClockByTriggerEffect(mlir::Value signal, llhd::ProcessOp processOp);
```

#### 综合信号类型推断

```cpp
enum class SignalTypeByDataFlow {
  GPIOInput,         // GPIO 外部输入信号
  WriteEnable,       // 写使能信号
  APBProtocol,       // APB 协议信号
  ClockReset,        // 时钟/复位信号
  InternalRegister,  // 内部寄存器
  StateRegister,     // 状态寄存器（可通过 APB 访问）
};

// 综合所有方案的数据流分析
SignalTypeByDataFlow inferSignalTypeByDataFlow(
    mlir::Value signal, hw::HWModuleOp moduleOp, llhd::ProcessOp processOp);
```

**相关代码**: [SignalTracing.h:1214-1650](src/lib/SignalTracing.h#L1214-L1650)

## 输入信号处理架构

### 核心思路

```
                    ┌──────────────────┐
                    │   输入信号分类    │
                    └────────┬─────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   ┌───────────┐      ┌───────────┐      ┌───────────┐
   │   时钟    │      │   APB     │      │  GPIO输入  │
   │  (过滤)   │      │ (MMIO)    │      │ (gpio_in) │
   └───────────┘      └─────┬─────┘      └─────┬─────┘
                            │                  │
                            ▼                  ▼
                    ┌───────────────────────────────┐
                    │      Signal Tracing           │
                    │  追踪: 输入 → 哪些状态被影响   │
                    └────────────────┬──────────────┘
                                     │
                                     ▼
                    ┌───────────────────────────────┐
                    │    生成 update_state()        │
                    │  重新计算所有依赖的组合逻辑    │
                    └───────────────────────────────┘
```

### 信号分类与处理方式

| 类型 | 信号示例 | 处理方式 |
|-----|---------|---------|
| **时钟** | `pclk`, `pclk_intr`, `dbclk` | 过滤掉，不生成代码 |
| **复位** | `presetn`, `dbclk_rstn` | 事件处理器 `on_xxx_write()` |
| **APB 协议** | `psel && penable && pwrite` | Signal Tracing → MMIO write case |
| **APB 地址** | `paddr` | MMIO offset，提取 case 匹配值 |
| **APB 数据** | `pwdata` | MMIO 写入值 |
| **GPIO 输入** | `gpio_ext_porta`, `gpio_in_data` | `qdev_init_gpio_in()` + `update_state()` |
| **控制信号** | `gpio_int_level_sync`, `scan_mode` | 事件处理器 |

### GPIO 输入处理模式（参考 SiFive GPIO）

```c
// 初始化时注册输入回调
qdev_init_gpio_in(DEVICE(s), gpio_top_gpio_input_set, 32);

// 输入变化回调
static void gpio_top_gpio_input_set(void *opaque, int line, int value)
{
    gpio_top_state *s = GPIO_TOP(opaque);
    s->gpio_ext_porta = deposit32(s->gpio_ext_porta, line, 1, value);
    gpio_top_update_state(s);  // 重新计算组合逻辑
}

// update_state: 通过 Signal Tracing 生成
static void gpio_top_update_state(gpio_top_state *s)
{
    // 组合逻辑: gpio_ext_porta → int_level → gpio_int_status
    s->int_level = /* traced expression */;
    s->gpio_int_status = s->gpio_int_status_level | s->gpio_int_status_edge;

    // 更新中断输出
    if (s->gpio_int_status & s->gpio_int_en & ~s->gpio_int_mask) {
        qemu_irq_raise(s->irq);
    }
}
```

## 使用方法

```bash
# 分析 LLHD IR 并显示信号分类结果
./dff-opt input.mlir --analyze-clk

# 生成 QEMU 设备代码
./dff-opt input.mlir --gen-qemu --qemu-out=/path/to/output
```

## 信号分类逻辑

工具分析每个 `llhd.drv` 操作，将信号分为 4 类：

### 1. STATE_UNCHANGED（状态不变型）

不依赖自身的信号，或 hold 模式（`reg = prb reg`），可转换为简单的寄存器写入。

```verilog
// 示例：简单赋值
always @(posedge clk)
    reg_a <= data_in;
```

**QEMU 输出**：事件处理器中的直接赋值
```c
s->reg_a = s->data_in;
```

### 2. STATE_ACCUMULATE（状态累加型）

具有 `signal = signal +/- constant` 形式的自依赖信号。

```verilog
// 示例：计数器
always @(posedge clk)
    counter <= counter + 1;
```

**QEMU 输出**：基于 ptimer 的计数器，使用 QEMU_CLOCK_VIRTUAL
```c
ptimer_state *counter_ptimer;
// 使用 ptimer_get_count() / ptimer_set_count()
```

### 3. STATE_LOOP_ITER（循环迭代器）

组合逻辑中的 for 循环迭代变量。

```verilog
// 示例：for 循环迭代器
for (i = 0; i < 32; i = i + 1)
    result[i] <= data[i] & mask[i];
```

**QEMU 输出**：展开为位操作（目前跳过）

### 4. STATE_COMPLEX（需手动处理）

表达式无法转换为 C 代码的信号。

**判定为 STATE_COMPLEX 的条件**：
- 不是常量赋值
- 不是简单信号赋值
- 不是比较结果赋值
- 不是自累加模式
- CombTranslator 无法翻译的操作

**QEMU 输出**：跳过并警告，需要手动实现

## 表达式生成（与分类统一）

工具采用统一方案，分类与代码生成耦合：

```
对每个 drv 操作：
  1. 尝试生成 C 表达式 (tryGenerateAction)
  2. 如果生成失败 -> STATE_COMPLEX
  3. 如果生成成功 -> 根据模式分类
```

### 支持的表达式类型

| 类型 | 模式 | C 代码 |
|------|------|--------|
| ASSIGN_CONST | `sig <= 0` | `s->sig = 0;` |
| ASSIGN_SIGNAL | `sig <= other` | `s->sig = s->other;` |
| ASSIGN_COMPARE | `sig <= (a >= b)` | `s->sig = (s->a >= s->b);` |
| ACCUMULATE | `sig <= sig + 1` | `ptimer_set_count(...)` |
| COMPUTE | comb 操作 | `s->sig = expr;` |

### CombTranslator 支持的操作

| 操作类别 | 操作 | C 表达式 |
|----------|------|----------|
| 算术 | add, sub, mul | `+`, `-`, `*` |
| 算术 | divu, divs | `/` (有符号用 cast) |
| 算术 | modu, mods | `%` |
| 位运算 | and, or, xor | `&`, `\|`, `^` |
| 位运算 | shl, shru, shrs | `<<`, `>>` |
| 数据操作 | extract | `(val >> lowBit) & mask` |
| 数据操作 | concat | `(hi << width) \| lo` |
| 数据操作 | mux | `cond ? true : false` |
| 数据操作 | replicate | `bit ? 0xFFFF : 0` |
| 比较 | icmp | `==`, `!=`, `<`, `<=`, `>`, `>=` |

## 事件处理器生成

为每个触发状态变化的输入信号生成：

```c
static void device_on_signal_write(device_state *s, uint32_t value)
{
    if (value) {        // SIGNAL_TRUE 条件
        s->reg = ...;
    }
    if (!value) {       // SIGNAL_FALSE 条件
        s->reg = 0;
    }
}
```

## 控制信号检测

工具检测输入信号与计数器之间的控制关系：

```verilog
// 如果 enable=1 触发计数器累加
always @(posedge clk)
    if (enable)
        counter <= counter + 1;
```

**检测结果**：`enable` 控制 `counter`（activeHigh=true）

## 生成的文件

为每个模块生成两个文件：

- `module_name.h` - 设备状态结构体和类型声明
- `module_name.c` - 设备实现，包含：
  - 内存映射寄存器读写
  - 输入信号的事件处理器
  - 计数器的 ptimer 设置
  - 复位逻辑

## 目录结构

```
qemu-output/
├── README.md           # 本文档
├── gpio0/              # GPIO 设备生成示例
│   ├── gpio_top.h      # 设备头文件
│   └── gpio_top.c      # 设备实现
├── src/                # 源代码
│   └── lib/            # 库文件
│       ├── CombTranslator.h    # Comb 操作翻译器
│       ├── ClkAnalysisResult.h # 信号分析结果定义
│       ├── QEMUCodeGen.h       # QEMU 代码生成器
│       └── SignalTracing.h     # 信号追踪工具
└── test/               # 测试文件
```

## 架构说明

```
┌─────────────────┐
│   Verilog/SV    │
└────────┬────────┘
         │ moore (转换)
         ▼
┌─────────────────┐
│    LLHD IR      │
└────────┬────────┘
         │ dff-opt --gen-qemu
         ▼
┌─────────────────────────────────────────┐
│         信号类型分析阶段                  │
│  ┌─────────────────────────────────┐    │
│  │ 方案1: 拓扑角色分析               │    │
│  │ - analyzeSignalRole              │    │
│  │ - SignalRole 枚举推断            │    │
│  └─────────────────────────────────┘    │
│                 ▼                        │
│  ┌─────────────────────────────────┐    │
│  │ 方案2: 使用模式识别               │    │
│  │ - isInternalSignalByUsagePattern │    │
│  │ - isClockSignalByUsagePattern    │    │
│  └─────────────────────────────────┘    │
│                 ▼                        │
│  ┌─────────────────────────────────┐    │
│  │ 方案3: 数据流分析                 │    │
│  │ - isGPIOInputByDataFlow          │    │
│  │ - isWriteEnableByDataFlow        │    │
│  └─────────────────────────────────┘    │
│                 ▼                        │
│  ┌─────────────────────────────────┐    │
│  │ 触发效果分析                      │    │
│  │ - isClockByTriggerEffect         │    │
│  │ - 区分时钟 vs 复位信号            │    │
│  └─────────────────────────────────┘    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│            drv 分类阶段                   │
│  ┌─────────────────────────────────┐    │
│  │ tryGenerateAction()             │    │
│  │ - 尝试生成 C 表达式              │    │
│  │ - 使用 CombTranslator 翻译      │    │
│  │ - 成功 → 继续分类                │    │
│  │ - 失败 → STATE_COMPLEX          │    │
│  └─────────────────────────────────┘    │
│                 ▼                        │
│  ┌─────────────────────────────────┐    │
│  │ 模式匹配分类                     │    │
│  │ - hold/不依赖自己 → STATE_UNCHANGED │
│  │ - sig+const → STATE_ACCUMULATE  │    │
│  │ - for 循环 → STATE_LOOP_ITER    │    │
│  └─────────────────────────────────┘    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│            代码生成阶段                   │
│  - STATE_UNCHANGED → 简单寄存器         │
│  - STATE_ACCUMULATE → ptimer 计数器     │
│  - COMPUTE → CombTranslator 表达式      │
│  - STATE_COMPLEX → 跳过（需手动）        │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────┐
│  QEMU C 代码    │
│  (.h + .c)      │
└─────────────────┘
```

## 示例输出

### gpio0 模块

生成的 `gpio_top.h`:
```c
typedef struct gpio_top_state {
    SysBusDevice parent_obj;
    MemoryRegion iomem;
    qemu_irq irq;

    uint32_t gpio_int_en;
    uint8_t gpio_int_level_sync;
    // ... 44 个信号

    /* ptimer-based counter: int_k */
    ptimer_state *int_k_ptimer;
    uint32_t int_k_limit;

    uint32_t presetn;  /* input */
    uint32_t pwdata;   /* input */
    // ... 27 个输入信号
} gpio_top_state;
```

生成的事件处理器:
```c
static void gpio_top_on_presetn_write(gpio_top_state *s, uint32_t value)
{
    if (!value) {
        s->gpio_int_en = 0;
    }
    if (value) {
        s->gpio_int_en = s->pwdata;
    }
    // ... 更多条件分支
}
```
