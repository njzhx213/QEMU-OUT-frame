# GPIO Top QEMU 设备代码

自动从 LLHD IR 生成的 QEMU GPIO 设备模拟代码

---

## 📁 文件列表

- `gpio_top.c` - 完整的 QEMU 设备实现 (571 行)
- `gpio_top.h` - 设备头文件
- `README.md` - 本说明文件

---

## 📊 生成统计

### 寄存器映射

**总计**: 11 个寄存器 (覆盖率 91.7%)

#### 可读写寄存器 (7 个)
| 地址 | 寄存器名 | 功能 |
|------|----------|------|
| 0x00 | gpio_sw_data | GPIO 数据寄存器 |
| 0x04 | gpio_sw_dir | GPIO 方向控制 |
| 0x30 | gpio_int_en | 中断使能 |
| 0x34 | gpio_int_mask | 中断屏蔽 |
| 0x38 | gpio_int_type | 中断类型 (边沿/电平) |
| 0x3c | gpio_int_pol | 中断极性 |
| 0x60 | gpio_int_level_sync | 中断电平同步 |

#### 只读寄存器 (4 个)
| 地址 | 寄存器名 | 功能 |
|------|----------|------|
| 0x40 | gpio_int_status | 中断状态寄存器 |
| 0x44 | gpio_raw_int_status | 原始中断状态 |
| 0x48 | gpio_debounce | 防抖配置 |
| 0x50 | gpio_ext_data | GPIO 外部输入数据 |

### MMIO 函数

- **gpio_top_read()**: 11 个 case (7 可读写 + 4 只读)
- **gpio_top_write()**: 7 个 case (只有可写寄存器)

---

## 🎯 功能特性

### GPIO 基本功能
✅ GPIO 输出控制 (通过 gpio_sw_data)
✅ GPIO 方向设置 (通过 gpio_sw_dir)
✅ GPIO 输入读取 (通过 gpio_ext_data)

### 中断功能
✅ 中断使能/屏蔽控制
✅ 中断类型配置 (边沿/电平)
✅ 中断极性设置
✅ 中断状态读取
✅ 原始中断状态读取

### 高级功能
✅ 防抖配置读取
✅ 中断电平同步控制
✅ GPIO 输入事件处理
✅ 中断输出信号

---

## 🔧 使用方法

### 1. 集成到 QEMU

将文件复制到 QEMU 源码目录:
```bash
cp gpio_top.c qemu/hw/gpio/
cp gpio_top.h qemu/include/hw/gpio/
```

### 2. 修改 QEMU 构建配置

在 `qemu/hw/gpio/meson.build` 中添加:
```meson
softmmu_ss.add(when: 'CONFIG_GPIO_TOP', if_true: files('gpio_top.c'))
```

### 3. 在设备树或命令行中使用

```bash
qemu-system-arm -device gpio_top,address=0x40000000
```

---

## 📝 代码结构

### 核心数据结构

```c
typedef struct gpio_top_state {
    SysBusDevice parent_obj;

    MemoryRegion iomem;
    qemu_irq irq;

    // 寄存器
    uint32_t gpio_sw_data;          // 0x00: GPIO 数据
    uint32_t gpio_sw_dir;           // 0x04: GPIO 方向
    uint32_t gpio_int_en;           // 0x30: 中断使能
    uint32_t gpio_int_mask;         // 0x34: 中断屏蔽
    uint32_t gpio_int_type;         // 0x38: 中断类型
    uint32_t gpio_int_pol;          // 0x3c: 中断极性
    uint32_t gpio_int_status;       // 0x40: 中断状态 (只读)
    uint32_t gpio_raw_int_status;   // 0x44: 原始中断 (只读)
    uint32_t gpio_debounce;         // 0x48: 防抖配置 (只读)
    uint32_t gpio_ext_data;         // 0x50: 外部输入 (只读)
    uint8_t  gpio_int_level_sync;   // 0x60: 电平同步

    // 内部状态
    ...
} gpio_top_state;
```

### 主要函数

```c
// MMIO 访问
static uint64_t gpio_top_read(void *opaque, hwaddr addr, unsigned size);
static void gpio_top_write(void *opaque, hwaddr addr, uint64_t value, unsigned size);

// 事件处理
static void gpio_top_on_gpio_int_level_sync_write(gpio_top_state *s, uint32_t value);
static void gpio_top_on_gpio_int_clr_write(gpio_top_state *s, uint32_t value);

// 设备生命周期
static void gpio_top_init(Object *obj);
static void gpio_top_realize(DeviceState *dev, Error **errp);
static void gpio_top_reset(DeviceState *dev);
```

---

## ⚙️ 提取技术细节

### 提取方法

本代码使用以下技术从 LLHD IR 自动提取:

1. **APB 写入寄存器提取**
   - 检测 `and(psel, penable, pwrite)` 模式
   - 追踪地址检查 `icmp(extract(paddr), const)`
   - 提取 `drv *_wen` 信号并去除后缀

2. **只读寄存器提取**
   - 直接搜索所有 `drv prdata` 操作
   - 使用 `traceToSignal()` 追踪值来源
   - 向上递归查找地址检查条件
   - 去除 `ri_` 前缀

3. **信号分类与过滤**
   - 自动过滤内部信号 (ri_*, *_wen, *_tmp, PROC.*, _ff*)
   - 分类时钟、复位、APB 协议信号
   - 识别 GPIO 输入信号

### 代码质量

- ✅ 0 个内部信号 (100% 过滤)
- ✅ 11 个真实寄存器
- ✅ 正确的读写权限
- ✅ 准确的地址映射
- ✅ 完整的事件处理

---

## 🐛 已知问题

### 1. 地址冲突

**问题**: gpio_int_clr 与 gpio_int_level_sync 共享地址 0x60

**原因**: LLHD IR 中两个寄存器使用相同的 APB 地址
- gpio_int_level_sync (i1) 使用 pwdata[0]
- gpio_int_clr (i32) 使用完整 pwdata

**当前处理**: 只保留 gpio_int_level_sync

**建议**: 查看原始 Verilog 设计意图

---

## 📈 性能与统计

### 代码规模
- 总行数: 571 行
- MMIO Read cases: 11 个
- MMIO Write cases: 7 个
- 事件处理器: 2 个

### 对比之前的实现
| 指标 | 之前 | 现在 | 改进 |
|------|------|------|------|
| case 总数 | 125 | 18 | -85.6% |
| 内部信号 | 80+ | 0 | -100% |
| 只读寄存器 | 0 | 4 | +400% |
| 寄存器覆盖率 | 58.3% | 91.7% | +33.4% |

---

## 📚 参考资料

### 相关文档
- [LLHD IR 规范](https://llhd.io/)
- [QEMU 设备开发文档](https://qemu.readthedocs.io/)
- [APB 协议规范](https://developer.arm.com/documentation/ihi0024/latest/)

### 源文件
- 输入: `verilog/gpio0_llhd.mlir`
- 工具: `qemu-transfer/build/dff-opt`
- 详细报告: `../EXTRACTION_TEST_RESULTS.md`

---

## 📞 联系与反馈

如有问题或建议,请参考项目文档或提交 issue。

---

**生成时间**: 2025-12-22
**工具版本**: dff-opt (LLHD to QEMU Converter)
**覆盖率**: 91.7% (11/12 寄存器)
