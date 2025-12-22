# APB 寄存器映射对照分析报告

## gpio0_llhd.mlir vs 生成的 QEMU 代码

生成日期: 2025-12-22
源文件: `workspace/verilog/gpio0_llhd.mlir`
生成文件: `workspace/qemu-output/gpio_top_fixed.c`

---

## 1. LLHD IR 中的完整寄存器映射

### APB 写入寄存器 (有 _wen 信号)

| 地址 (字) | 字节地址 | 寄存器名 | 位宽 | 状态 |
|----------|----------|----------|------|------|
| 0 | 0x00 | gpio_sw_data | i32 | ✅ 已提取 |
| 1 | 0x04 | gpio_sw_dir | i32 | ✅ 已提取 |
| 12 | 0x30 | gpio_int_en | i32 | ✅ 已提取 |
| 13 | 0x34 | gpio_int_mask | i32 | ✅ 已提取 |
| 14 | 0x38 | gpio_int_type | i32 | ✅ 已提取 |
| 15 | 0x3c | gpio_int_pol | i32 | ✅ 已提取 |
| 24 | 0x60 | gpio_int_level_sync | i1 | ✅ 已提取 |
| 24 | 0x60 | gpio_int_clr | i32 | ⚠️ **地址冲突,未提取** |

### APB 只读寄存器 (无 _wen 信号,仅 prdata 返回)

| 地址 (字) | 字节地址 | 寄存器名 | 位宽 | 状态 |
|----------|----------|----------|------|------|
| 16 | 0x40 | gpio_debounce | i32 | ❌ **未提取** (只读?) |
| 17 | 0x44 | gpio_int_status | i32 | ❌ **未提取** (只读) |
| 18 | 0x48 | gpio_raw_int_status | i32 | ❌ **未提取** (只读) |
| 20 | 0x50 | gpio_ext_data | i32 | ❌ **未提取** (GPIO 输入) |
| 24 | 0x60 | ri_gpio_int_level_sync | i32 | 📝 备注: prdata 返回 |

---

## 2. 生成的 QEMU 代码寄存器映射

### MMIO Read (7 个 case)

```c
case 0x00:  /* gpio_sw_data */
case 0x04:  /* gpio_sw_dir */
case 0x30:  /* gpio_int_en */
case 0x34:  /* gpio_int_mask */
case 0x38:  /* gpio_int_type */
case 0x3c:  /* gpio_int_pol */
case 0x60:  /* gpio_int_level_sync */
```

### MMIO Write (7 个 case)

```c
case 0x00:  /* gpio_sw_data */
case 0x04:  /* gpio_sw_dir */
case 0x30:  /* gpio_int_en */
case 0x34:  /* gpio_int_mask */
case 0x38:  /* gpio_int_type */
case 0x3c:  /* gpio_int_pol */
case 0x60:  /* gpio_int_level_sync */
```

---

## 3. 问题总结

### ✅ 已解决的问题

1. **内部信号过滤**: 所有 `ri_*`, `*_wen`, `*_tmp`, `PROC.*` 内部信号已被过滤
2. **真实寄存器名**: 成功去除 `_wen` 后缀,使用真实寄存器名
3. **APB 地址提取**: 成功从 LLHD IR 提取真实 APB 地址
4. **地址符号扩展 bug**: 修复了 `getSExtValue()` → `getZExtValue()`,地址从 0xffffffe0 修正为 0x60

### ⚠️ 仍存在的问题

#### 问题 1: 地址冲突 (0x60)

**现象**: 两个寄存器共享同一地址
- `gpio_int_level_sync` (i1, 使用 pwdata[0])
- `gpio_int_clr` (i32, 使用完整 pwdata, 有自动清零特性)

**当前行为**: 只保留第一个 (`gpio_int_level_sync`)

**LLHD IR 证据**:
```mlir
// 写入逻辑 - 两个 process 都检查地址 24
%49 = comb.icmp eq %48, %c-8_i5 : i5  // -8 (i5) = 24 (无符号)
llhd.drv %gpio_int_level_sync_wen, %true after %1

%49 = comb.icmp eq %48, %c-8_i5 : i5
llhd.drv %gpio_int_clr_wen, %true after %1

// 实际写入
// gpio_int_level_sync: 从 pwdata[0] 提取 1 位
%56 = comb.extract %55 from 0 : (i32) -> i1
llhd.drv %gpio_int_level_sync, %56 after %0

// gpio_int_clr: 使用完整 pwdata,不写入时自动清零
llhd.drv %gpio_int_clr, %44 after %1        // 写入
llhd.drv %gpio_int_clr, %c0_i32 after %1    // 不写入时清零
```

**可能的解决方案**:
- **方案 A**: 合并为一个 32 位寄存器,bit 0 用于 level_sync,其余用于 int_clr
- **方案 B**: 保持现状,添加警告注释
- **方案 C**: 需要查看原始 Verilog 代码确认设计意图

#### 问题 2: 遗漏只读寄存器

**原因**: 当前提取逻辑只追踪 APB 写入条件 (`psel && penable && pwrite`)

**遗漏的寄存器**:
1. **gpio_int_status** (0x44): 中断状态寄存器 (只读)
2. **gpio_raw_int_status** (0x48): 原始中断状态 (只读)
3. **gpio_ext_data** (0x50): GPIO 外部输入数据 (只读)
4. **gpio_debounce** (0x40): 防抖配置 (可能可读写?)

**LLHD IR 证据** (prdata 读取逻辑):
```mlir
%67 = comb.icmp ceq %62, %c17_i32 : i32
// 返回 gpio_int_status

%78 = comb.icmp ceq %62, %c18_i32 : i32
// 返回 gpio_raw_int_status

%84 = comb.icmp ceq %62, %c20_i32 : i32
// 返回 gpio_ext_data

%76 = comb.icmp ceq %62, %c16_i32 : i32
// 返回 gpio_debounce
```

**建议**: 需要添加 prdata 读取逻辑的提取,补充只读寄存器

---

## 4. 数据完整性验证

### 提取准确率

| 类别 | LLHD IR 中数量 | 已提取 | 准确率 |
|-----|---------------|--------|--------|
| **可写寄存器** | 8 个 | 7 个 | 87.5% |
| **只读寄存器** | 4 个 | 0 个 | 0% |
| **总计** | 12 个 | 7 个 | 58.3% |

### 地址覆盖率

```
LLHD IR 地址: 0x00 0x04 0x30 0x34 0x38 0x3c 0x40 0x44 0x48 0x50 0x60
已提取地址:   ✅   ✅   ✅   ✅   ✅   ✅   ❌   ❌   ❌   ❌   ✅(冲突)
```

---

## 5. 建议的后续工作

### 优先级 1: 修复地址冲突

需要分析原始 Verilog 代码,确定 0x60 地址的设计意图:
- 是两个独立寄存器?
- 还是一个寄存器的不同字段?

### 优先级 2: 添加只读寄存器提取

扩展 `extractAPBRegisterMappings()` 功能:
1. 分析 prdata 赋值逻辑
2. 提取只读寄存器的地址和名称
3. 标记为只读 (`isWritable = false`)

### 优先级 3: 验证生成代码的功能正确性

1. 检查中断状态寄存器是否影响功能
2. 验证 GPIO 输入是否需要通过 MMIO 访问
3. 确认防抖配置寄存器的重要性

---

## 6. 结论

**当前状态**:
- ✅ 核心可写寄存器提取成功 (7/8)
- ✅ 地址提取正确 (修复符号扩展 bug)
- ✅ 内部信号完全过滤
- ⚠️ 存在地址冲突和遗漏只读寄存器

**总体评价**:
对于可写寄存器的提取已经达到 87.5% 准确率,主要功能寄存器 (SW_DATA, SW_DIR, INT_EN, INT_MASK, INT_TYPE, INT_POL) 全部正确提取。只读寄存器的缺失可能影响中断功能和 GPIO 输入读取。

**代码可用性**:
生成的代码可以用于基本的 GPIO 控制(输出、方向、中断使能/屏蔽/类型/极性),但缺少中断状态读取和 GPIO 输入读取功能。
