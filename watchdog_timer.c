/*
 * Auto-generated QEMU device: watchdog_timer
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
#include "watchdog_timer.h"

/*
 * ptimer 到期回调 - 计数器归零时触发
 */
static void watchdog_timer_timer_tick(void *opaque)
{
    watchdog_timer_state *s = opaque;
    /* 触发中断 */
    qemu_irq_raise(s->irq);
}

/*
 * 读取 counter 计数器
 * 使用 ptimer 获取当前值
 */
static uint32_t watchdog_timer_get_counter(watchdog_timer_state *s)
{
    return ptimer_get_count(s->counter_ptimer);
}

/*
 * 写入 counter 计数器（加载新值）
 */
static void watchdog_timer_set_counter(watchdog_timer_state *s, uint32_t value)
{
    ptimer_transaction_begin(s->counter_ptimer);
    ptimer_set_count(s->counter_ptimer, value);
    ptimer_transaction_commit(s->counter_ptimer);
}

/*
 * 设置 counter 计数器上限/reload值
 */
static void watchdog_timer_set_counter_limit(watchdog_timer_state *s, uint32_t limit)
{
    s->counter_limit = limit;
    ptimer_transaction_begin(s->counter_ptimer);
    ptimer_set_limit(s->counter_ptimer, limit, 1);
    ptimer_transaction_commit(s->counter_ptimer);
}

/*
 * 启动 counter 计数器
 */
static void watchdog_timer_start_counter(watchdog_timer_state *s)
{
    ptimer_transaction_begin(s->counter_ptimer);
    ptimer_run(s->counter_ptimer, 0);  /* 0 = 周期模式 */
    ptimer_transaction_commit(s->counter_ptimer);
}

/*
 * 停止 counter 计数器
 */
static void watchdog_timer_stop_counter(watchdog_timer_state *s)
{
    ptimer_transaction_begin(s->counter_ptimer);
    ptimer_stop(s->counter_ptimer);
    ptimer_transaction_commit(s->counter_ptimer);
}

/*
 * 获取派生信号 warning_threshold
 * 计算: timeout_val >> 1
 */
static uint32_t watchdog_timer_get_warning_threshold(watchdog_timer_state *s)
{
    return s->timeout_val >> 1;
}

/*
 * Event Handlers - triggered by input signal writes
 */

/*
 * Handler for enable signal changes
 * Controls ptimer: counter(active-high)
 */
static void watchdog_timer_on_enable_write(watchdog_timer_state *s, uint32_t value)
{
    if (value) {
        watchdog_timer_start_counter(s);  /* 启动 ptimer */
        if (value) {
        }
    } else {
        watchdog_timer_stop_counter(s);  /* 停止 ptimer */
        watchdog_timer_set_counter(s, 0);
        s->wdt_reset = 0;
        s->wdt_warning = 0;
    }
}

/*
 * Handler for feed signal changes
 * Controls ptimer: counter(active-low)
 */
static void watchdog_timer_on_feed_write(watchdog_timer_state *s, uint32_t value)
{
    if (value) {
        watchdog_timer_stop_counter(s);  /* 停止 ptimer */
        watchdog_timer_set_counter(s, 0);
        s->wdt_reset = 0;
        s->wdt_warning = 0;
    } else {
        watchdog_timer_start_counter(s);  /* 启动 ptimer */
        if (watchdog_timer_get_counter(s) >= s->timeout_val) {
            s->wdt_reset = 1;
        }
        if (watchdog_timer_get_counter(s) < s->timeout_val) {
            /* counter accumulate handled by ptimer */
            s->wdt_warning = (watchdog_timer_get_counter(s) >= watchdog_timer_get_warning_threshold(s));
        }
    }
}

static uint64_t watchdog_timer_read(void *opaque, hwaddr addr, unsigned size)
{
    watchdog_timer_state *s = opaque;
    uint64_t value = 0;

    switch (addr) {
    case 0x00:  /* wdt_reset */
        value = s->wdt_reset;
        break;
    case 0x04:  /* counter */
        value = watchdog_timer_get_counter(s);
        break;
    case 0x08:  /* wdt_warning */
        value = s->wdt_warning;
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR, "watchdog_timer: bad read at 0x%" HWADDR_PRIx "\n", addr);
    }
    return value;
}

static void watchdog_timer_write(void *opaque, hwaddr addr,
                              uint64_t value, unsigned size)
{
    watchdog_timer_state *s = opaque;

    switch (addr) {
    case 0x00:  /* wdt_reset */
        s->wdt_reset = value;
        break;
    case 0x04:  /* counter */
        watchdog_timer_set_counter(s, value);
        break;
    case 0x08:  /* wdt_warning */
        s->wdt_warning = value;
        break;
    case 0x0c:  /* enable (input) */
        s->enable = value;
        watchdog_timer_on_enable_write(s, value);
        break;
    case 0x10:  /* feed (input) */
        s->feed = value;
        watchdog_timer_on_feed_write(s, value);
        break;
    case 0x14:  /* timeout_val (input) */
        s->timeout_val = value;
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR, "watchdog_timer: bad write at 0x%" HWADDR_PRIx "\n", addr);
    }
}

static const MemoryRegionOps watchdog_timer_ops = {
    .read = watchdog_timer_read,
    .write = watchdog_timer_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
};

static void watchdog_timer_init(Object *obj)
{
    watchdog_timer_state *s = WATCHDOG_TIMER(obj);
    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);

    memory_region_init_io(&s->iomem, obj, &watchdog_timer_ops, s,
                          TYPE_WATCHDOG_TIMER, 0x1000);
    sysbus_init_mmio(sbd, &s->iomem);
    sysbus_init_irq(sbd, &s->irq);
}

static void watchdog_timer_realize(DeviceState *dev, Error **errp)
{
    watchdog_timer_state *s = WATCHDOG_TIMER(dev);

    /*
     * 创建 ptimer: counter
     * 使用 QEMU_CLOCK_VIRTUAL - 在 icount 模式下用 instruction counter
     *                          在普通模式下用 host timer
     */
    s->counter_ptimer = ptimer_init(watchdog_timer_timer_tick, s,
                                                PTIMER_POLICY_DEFAULT);
    ptimer_transaction_begin(s->counter_ptimer);
    /* 设置频率 (每秒多少次) - 根据 HDL 时钟频率调整 */
    ptimer_set_freq(s->counter_ptimer, 1000000);  /* 1MHz */
    ptimer_transaction_commit(s->counter_ptimer);

}

static void watchdog_timer_reset(DeviceState *dev)
{
    watchdog_timer_state *s = WATCHDOG_TIMER(dev);

    s->wdt_reset = 0;
    /* 重置 ptimer: counter */
    ptimer_transaction_begin(s->counter_ptimer);
    ptimer_stop(s->counter_ptimer);
    ptimer_set_count(s->counter_ptimer, 0);
    ptimer_transaction_commit(s->counter_ptimer);
    s->counter_limit = 0;
    s->wdt_warning = 0;
    qemu_irq_lower(s->irq);
}

static void watchdog_timer_class_init(ObjectClass *oc, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(oc);
    dc->realize = watchdog_timer_realize;
    dc->reset = watchdog_timer_reset;
}

static const TypeInfo watchdog_timer_info = {
    .name = TYPE_WATCHDOG_TIMER,
    .parent = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(watchdog_timer_state),
    .instance_init = watchdog_timer_init,
    .class_init = watchdog_timer_class_init,
};

static void watchdog_timer_register_types(void)
{
    type_register_static(&watchdog_timer_info);
}

type_init(watchdog_timer_register_types)
