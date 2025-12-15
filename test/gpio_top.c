/*
 * Auto-generated QEMU device: gpio_top
 *
 * Uses ptimer with QEMU_CLOCK_VIRTUAL.
 * - In icount mode: virtual clock driven by instruction counter
 * - In real-time mode: virtual clock driven by host timer
 */

#include "qemu/osdep.h"
#include "hw/irq.h"
#include "hw/qdev-properties.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "gpio_top.h"

/*
 * ptimer 到期回调 - 计数器归零时触发
 */
static void gpio_top_timer_tick(void *opaque)
{
    gpio_top_state *s = opaque;
    /* 触发中断 */
    qemu_irq_raise(s->irq);
}

/*
 * 读取 int_k 计数器
 * 使用 ptimer 获取当前值
 */
static uint32_t gpio_top_get_int_k(gpio_top_state *s)
{
    return ptimer_get_count(s->int_k_ptimer);
}

/*
 * 写入 int_k 计数器（加载新值）
 */
static void gpio_top_set_int_k(gpio_top_state *s, uint32_t value)
{
    ptimer_transaction_begin(s->int_k_ptimer);
    ptimer_set_count(s->int_k_ptimer, value);
    ptimer_transaction_commit(s->int_k_ptimer);
}

/*
 * 设置 int_k 计数器上限/reload值
 */
static void gpio_top_set_int_k_limit(gpio_top_state *s, uint32_t limit)
{
    s->int_k_limit = limit;
    ptimer_transaction_begin(s->int_k_ptimer);
    ptimer_set_limit(s->int_k_ptimer, limit, 1);
    ptimer_transaction_commit(s->int_k_ptimer);
}

/*
 * 启动 int_k 计数器
 */
static void gpio_top_start_int_k(gpio_top_state *s)
{
    ptimer_transaction_begin(s->int_k_ptimer);
    ptimer_run(s->int_k_ptimer, 0);  /* 0 = 周期模式 */
    ptimer_transaction_commit(s->int_k_ptimer);
}

/*
 * 停止 int_k 计数器
 */
static void gpio_top_stop_int_k(gpio_top_state *s)
{
    ptimer_transaction_begin(s->int_k_ptimer);
    ptimer_stop(s->int_k_ptimer);
    ptimer_transaction_commit(s->int_k_ptimer);
}

/*
 * Event Handlers - triggered by input signal writes
 */

/*
 * Handler for presetn signal changes
 */
static void gpio_top_on_presetn_write(gpio_top_state *s, uint32_t value)
{
    if (!value) {
        s->gpio_int_en = 0;
    }
    if (value) {
        if (value) {
            s->gpio_int_en = s->pwdata;
        }
    }
    if (!value) {
        s->gpio_int_mask = 0;
    }
    if (value) {
        if (value) {
            s->gpio_int_mask = s->pwdata;
        }
    }
    if (!value) {
        s->gpio_int_type = 0;
    }
    if (value) {
        if (value) {
            s->gpio_int_type = s->pwdata;
        }
    }
    if (!value) {
        s->gpio_int_pol = 0;
    }
    if (value) {
        if (value) {
            s->gpio_int_pol = s->pwdata;
        }
    }
    if (!value) {
        s->gpio_int_level_sync = 0;
    }
    if (value) {
        if (value) {
            /* complex expression */
        }
    }
    if (!value) {
        s->gpio_sw_data = 0;
    }
    if (value) {
        if (value) {
            s->gpio_sw_data = s->pwdata;
        }
    }
    if (!value) {
        s->gpio_sw_dir = 0;
    }
    if (value) {
        if (value) {
            s->gpio_sw_dir = s->pwdata;
        }
    }
    if (!value) {
        s->prdata = 0;
    }
    if (!value) {
        s->gpio_int_clk_en = 0;
    }
    if (value) {
        s->gpio_int_clk_en = s->gpio_int_clk_en_tmp;
    }
    if (!value) {
        s->int_level_ff1 = 0;
    }
    if (value) {
        s->int_level_ff1 = s->int_level;
    }
    if (!value) {
        s->gpio_int_status_edge = 0;
    }
    if (value) {
        gpio_top_set_int_k(s, 0);
    }
    if (!value) {
        s->SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff1 = 0;
    }
    if (value) {
        s->SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff1 = s->int_level_sync_in;
    }
    if (!value) {
        s->SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff2 = 0;
    }
    if (value) {
        s->SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff2 = s->SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff1;
    }
}

/*
 * Handler for gpio_int_level_sync signal changes
 */
static void gpio_top_on_gpio_int_level_sync_write(gpio_top_state *s, uint32_t value)
{
    if (value) {
        /* complex expression */
    }
    if (!value) {
        /* complex expression */
    }
}

static uint64_t gpio_top_read(void *opaque, hwaddr addr, unsigned size)
{
    gpio_top_state *s = opaque;
    uint64_t value = 0;

    switch (addr) {
    case 0x00:  /* gpio_raw_int_status */
        value = s->gpio_raw_int_status;
        break;
    case 0x04:  /* gpio_int_level_sync */
        value = s->gpio_int_level_sync;
        break;
    case 0x08:  /* gpio_int_en_wen */
        value = s->gpio_int_en_wen;
        break;
    case 0x0c:  /* ri_gpio_sw_data */
        value = s->ri_gpio_sw_data;
        break;
    case 0x10:  /* gpio_int_pol */
        value = s->gpio_int_pol;
        break;
    case 0x14:  /* gpio_int_clk_en */
        value = s->gpio_int_clk_en;
        break;
    case 0x18:  /* gpio_debounce_wen */
        value = s->gpio_debounce_wen;
        break;
    case 0x1c:  /* gpio_int_level_sync_wen */
        value = s->gpio_int_level_sync_wen;
        break;
    case 0x20:  /* gpio_int_type */
        value = s->gpio_int_type;
        break;
    case 0x24:  /* gpio_int_clr */
        value = s->gpio_int_clr;
        break;
    case 0x28:  /* int_k */
        value = gpio_top_get_int_k(s);
        break;
    case 0x2c:  /* SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff2 */
        value = s->SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff2;
        break;
    case 0x30:  /* prdata */
        value = s->prdata;
        break;
    case 0x34:  /* gpio_int_clr_wen */
        value = s->gpio_int_clr_wen;
        break;
    case 0x38:  /* gpio_int_status_edge */
        value = s->gpio_int_status_edge;
        break;
    case 0x3c:  /* gpio_sw_data */
        value = s->gpio_sw_data;
        break;
    case 0x40:  /* int_edge_out */
        value = s->int_edge_out;
        break;
    case 0x44:  /* gpio_int_mask_wen */
        value = s->gpio_int_mask_wen;
        break;
    case 0x48:  /* gpio_sw_dir */
        value = s->gpio_sw_dir;
        break;
    case 0x4c:  /* ri_gpio_sw_dir */
        value = s->ri_gpio_sw_dir;
        break;
    case 0x50:  /* gpio_int_pol_wen */
        value = s->gpio_int_pol_wen;
        break;
    case 0x54:  /* SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff1 */
        value = s->SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff1;
        break;
    case 0x58:  /* gpio_int_status_level */
        value = s->gpio_int_status_level;
        break;
    case 0x5c:  /* int_level_ff1 */
        value = s->int_level_ff1;
        break;
    case 0x60:  /* gpio_rx_data_int */
        value = s->gpio_rx_data_int;
        break;
    case 0x64:  /* gpio_int_mask */
        value = s->gpio_int_mask;
        break;
    case 0x68:  /* gpio_int_en */
        value = s->gpio_int_en;
        break;
    case 0x6c:  /* unnamed */
        value = s->unnamed;
        break;
    case 0x70:  /* gpio_sw_data_wen */
        value = s->gpio_sw_data_wen;
        break;
    case 0x74:  /* gpio_int_type_wen */
        value = s->gpio_int_type_wen;
        break;
    case 0x78:  /* int_level_sync_in */
        value = s->int_level_sync_in;
        break;
    case 0x7c:  /* gpio_sw_dir_wen */
        value = s->gpio_sw_dir_wen;
        break;
    case 0x80:  /* ri_gpio_ext_data */
        value = s->ri_gpio_ext_data;
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR, "gpio_top: bad read at 0x%" HWADDR_PRIx "\n", addr);
    }
    return value;
}

static void gpio_top_write(void *opaque, hwaddr addr,
                              uint64_t value, unsigned size)
{
    gpio_top_state *s = opaque;

    switch (addr) {
    case 0x00:  /* gpio_raw_int_status */
        s->gpio_raw_int_status = value;
        break;
    case 0x04:  /* gpio_int_level_sync */
        s->gpio_int_level_sync = value;
        gpio_top_on_gpio_int_level_sync_write(s, value);
        break;
    case 0x08:  /* gpio_int_en_wen */
        s->gpio_int_en_wen = value;
        break;
    case 0x0c:  /* ri_gpio_sw_data */
        s->ri_gpio_sw_data = value;
        break;
    case 0x10:  /* gpio_int_pol */
        s->gpio_int_pol = value;
        break;
    case 0x14:  /* gpio_int_clk_en */
        s->gpio_int_clk_en = value;
        break;
    case 0x18:  /* gpio_debounce_wen */
        s->gpio_debounce_wen = value;
        break;
    case 0x1c:  /* gpio_int_level_sync_wen */
        s->gpio_int_level_sync_wen = value;
        break;
    case 0x20:  /* gpio_int_type */
        s->gpio_int_type = value;
        break;
    case 0x24:  /* gpio_int_clr */
        s->gpio_int_clr = value;
        break;
    case 0x28:  /* int_k */
        gpio_top_set_int_k(s, value);
        break;
    case 0x2c:  /* SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff2 */
        s->SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff2 = value;
        break;
    case 0x30:  /* prdata */
        s->prdata = value;
        break;
    case 0x34:  /* gpio_int_clr_wen */
        s->gpio_int_clr_wen = value;
        break;
    case 0x38:  /* gpio_int_status_edge */
        s->gpio_int_status_edge = value;
        break;
    case 0x3c:  /* gpio_sw_data */
        s->gpio_sw_data = value;
        break;
    case 0x40:  /* int_edge_out */
        s->int_edge_out = value;
        break;
    case 0x44:  /* gpio_int_mask_wen */
        s->gpio_int_mask_wen = value;
        break;
    case 0x48:  /* gpio_sw_dir */
        s->gpio_sw_dir = value;
        break;
    case 0x4c:  /* ri_gpio_sw_dir */
        s->ri_gpio_sw_dir = value;
        break;
    case 0x50:  /* gpio_int_pol_wen */
        s->gpio_int_pol_wen = value;
        break;
    case 0x54:  /* SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff1 */
        s->SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff1 = value;
        break;
    case 0x58:  /* gpio_int_status_level */
        s->gpio_int_status_level = value;
        break;
    case 0x5c:  /* int_level_ff1 */
        s->int_level_ff1 = value;
        break;
    case 0x60:  /* gpio_rx_data_int */
        s->gpio_rx_data_int = value;
        break;
    case 0x64:  /* gpio_int_mask */
        s->gpio_int_mask = value;
        break;
    case 0x68:  /* gpio_int_en */
        s->gpio_int_en = value;
        break;
    case 0x6c:  /* unnamed */
        s->unnamed = value;
        break;
    case 0x70:  /* gpio_sw_data_wen */
        s->gpio_sw_data_wen = value;
        break;
    case 0x74:  /* gpio_int_type_wen */
        s->gpio_int_type_wen = value;
        break;
    case 0x78:  /* int_level_sync_in */
        s->int_level_sync_in = value;
        break;
    case 0x7c:  /* gpio_sw_dir_wen */
        s->gpio_sw_dir_wen = value;
        break;
    case 0x80:  /* ri_gpio_ext_data */
        s->ri_gpio_ext_data = value;
        break;
    case 0x84:  /* dbclk (input) */
        s->dbclk = value;
        break;
    case 0x88:  /* dbclk_rstn (input) */
        s->dbclk_rstn = value;
        break;
    case 0x8c:  /* gpio_debounce (input) */
        s->gpio_debounce = value;
        break;
    case 0x90:  /* gpio_ext_data (input) */
        s->gpio_ext_data = value;
        break;
    case 0x94:  /* gpio_ext_porta (input) */
        s->gpio_ext_porta = value;
        break;
    case 0x98:  /* gpio_in_data (input) */
        s->gpio_in_data = value;
        break;
    case 0x9c:  /* gpio_int_status (input) */
        s->gpio_int_status = value;
        break;
    case 0xa0:  /* gpio_rx_data (input) */
        s->gpio_rx_data = value;
        break;
    case 0xa4:  /* paddr (input) */
        s->paddr = value;
        break;
    case 0xa8:  /* pclk (input) */
        s->pclk = value;
        break;
    case 0xac:  /* pclk_int (input) */
        s->pclk_int = value;
        break;
    case 0xb0:  /* pclk_intr (input) */
        s->pclk_intr = value;
        break;
    case 0xb4:  /* penable (input) */
        s->penable = value;
        break;
    case 0xb8:  /* presetn (input) */
        s->presetn = value;
        gpio_top_on_presetn_write(s, value);
        break;
    case 0xbc:  /* psel (input) */
        s->psel = value;
        break;
    case 0xc0:  /* pwdata (input) */
        s->pwdata = value;
        break;
    case 0xc4:  /* pwrite (input) */
        s->pwrite = value;
        break;
    case 0xc8:  /* scan_mode (input) */
        s->scan_mode = value;
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR, "gpio_top: bad write at 0x%" HWADDR_PRIx "\n", addr);
    }
}

static const MemoryRegionOps gpio_top_ops = {
    .read = gpio_top_read,
    .write = gpio_top_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
};

static void gpio_top_init(Object *obj)
{
    gpio_top_state *s = GPIO_TOP(obj);
    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);

    memory_region_init_io(&s->iomem, obj, &gpio_top_ops, s,
                          TYPE_GPIO_TOP, 0x1000);
    sysbus_init_mmio(sbd, &s->iomem);
    sysbus_init_irq(sbd, &s->irq);
}

static void gpio_top_realize(DeviceState *dev, Error **errp)
{
    gpio_top_state *s = GPIO_TOP(dev);

    /*
     * 创建 ptimer: int_k
     * 使用 QEMU_CLOCK_VIRTUAL - 在 icount 模式下用 instruction counter
     *                          在普通模式下用 host timer
     */
    s->int_k_ptimer = ptimer_init(gpio_top_timer_tick, s,
                                                PTIMER_POLICY_DEFAULT);
    ptimer_transaction_begin(s->int_k_ptimer);
    /* 设置频率 (每秒多少次) - 根据 HDL 时钟频率调整 */
    ptimer_set_freq(s->int_k_ptimer, 1000000);  /* 1MHz */
    ptimer_transaction_commit(s->int_k_ptimer);

}

static void gpio_top_reset(DeviceState *dev)
{
    gpio_top_state *s = GPIO_TOP(dev);

    s->gpio_raw_int_status = 0;
    s->gpio_int_level_sync = 0;
    s->gpio_int_en_wen = 0;
    s->ri_gpio_sw_data = 0;
    s->gpio_int_pol = 0;
    s->gpio_int_clk_en = 0;
    s->gpio_debounce_wen = 0;
    s->gpio_int_level_sync_wen = 0;
    s->gpio_int_type = 0;
    s->gpio_int_clr = 0;
    /* 重置 ptimer: int_k */
    ptimer_transaction_begin(s->int_k_ptimer);
    ptimer_stop(s->int_k_ptimer);
    ptimer_set_count(s->int_k_ptimer, 0);
    ptimer_transaction_commit(s->int_k_ptimer);
    s->int_k_limit = 0;
    s->SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff2 = 0;
    s->prdata = 0;
    s->gpio_int_clr_wen = 0;
    s->gpio_int_status_edge = 0;
    s->gpio_sw_data = 0;
    s->int_edge_out = 0;
    s->gpio_int_mask_wen = 0;
    s->gpio_sw_dir = 0;
    s->ri_gpio_sw_dir = 0;
    s->gpio_int_pol_wen = 0;
    s->SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff1 = 0;
    s->gpio_int_status_level = 0;
    s->int_level_ff1 = 0;
    s->gpio_rx_data_int = 0;
    s->gpio_int_mask = 0;
    s->gpio_int_en = 0;
    s->unnamed = 0;
    s->gpio_sw_data_wen = 0;
    s->gpio_int_type_wen = 0;
    s->int_level_sync_in = 0;
    s->gpio_sw_dir_wen = 0;
    s->ri_gpio_ext_data = 0;
    qemu_irq_lower(s->irq);
}

static void gpio_top_class_init(ObjectClass *oc, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(oc);
    dc->realize = gpio_top_realize;
    dc->reset = gpio_top_reset;
}

static const TypeInfo gpio_top_info = {
    .name = TYPE_GPIO_TOP,
    .parent = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(gpio_top_state),
    .instance_init = gpio_top_init,
    .class_init = gpio_top_class_init,
};

static void gpio_top_register_types(void)
{
    type_register_static(&gpio_top_info);
}

type_init(gpio_top_register_types)
