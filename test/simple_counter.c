/*
 * Auto-generated QEMU device: simple_counter
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
#include "simple_counter.h"

/*
 * ptimer 到期回调 - 计数器归零时触发
 */
static void simple_counter_timer_tick(void *opaque)
{
    simple_counter_state *s = opaque;
    /* 触发中断 */
    qemu_irq_raise(s->irq);
}

/*
 * 读取 count 计数器
 * 使用 ptimer 获取当前值
 */
static uint8_t simple_counter_get_count(simple_counter_state *s)
{
    return ptimer_get_count(s->count_ptimer);
}

/*
 * 写入 count 计数器（加载新值）
 */
static void simple_counter_set_count(simple_counter_state *s, uint8_t value)
{
    ptimer_transaction_begin(s->count_ptimer);
    ptimer_set_count(s->count_ptimer, value);
    ptimer_transaction_commit(s->count_ptimer);
}

/*
 * 设置 count 计数器上限/reload值
 */
static void simple_counter_set_count_limit(simple_counter_state *s, uint8_t limit)
{
    s->count_limit = limit;
    ptimer_transaction_begin(s->count_ptimer);
    ptimer_set_limit(s->count_ptimer, limit, 1);
    ptimer_transaction_commit(s->count_ptimer);
}

/*
 * 启动 count 计数器
 */
static void simple_counter_start_count(simple_counter_state *s)
{
    ptimer_transaction_begin(s->count_ptimer);
    ptimer_run(s->count_ptimer, 0);  /* 0 = 周期模式 */
    ptimer_transaction_commit(s->count_ptimer);
}

/*
 * 停止 count 计数器
 */
static void simple_counter_stop_count(simple_counter_state *s)
{
    ptimer_transaction_begin(s->count_ptimer);
    ptimer_stop(s->count_ptimer);
    ptimer_transaction_commit(s->count_ptimer);
}

static uint64_t simple_counter_read(void *opaque, hwaddr addr, unsigned size)
{
    simple_counter_state *s = opaque;
    uint64_t value = 0;

    switch (addr) {
    case 0x00:  /* count */
        value = simple_counter_get_count(s);
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR, "simple_counter: bad read at 0x%" HWADDR_PRIx "\n", addr);
    }
    return value;
}

static void simple_counter_write(void *opaque, hwaddr addr,
                              uint64_t value, unsigned size)
{
    simple_counter_state *s = opaque;

    switch (addr) {
    case 0x00:  /* count */
        simple_counter_set_count(s, value);
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR, "simple_counter: bad write at 0x%" HWADDR_PRIx "\n", addr);
    }
}

static const MemoryRegionOps simple_counter_ops = {
    .read = simple_counter_read,
    .write = simple_counter_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
};

static void simple_counter_init(Object *obj)
{
    simple_counter_state *s = SIMPLE_COUNTER(obj);
    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);

    memory_region_init_io(&s->iomem, obj, &simple_counter_ops, s,
                          TYPE_SIMPLE_COUNTER, 0x1000);
    sysbus_init_mmio(sbd, &s->iomem);
    sysbus_init_irq(sbd, &s->irq);
}

static void simple_counter_realize(DeviceState *dev, Error **errp)
{
    simple_counter_state *s = SIMPLE_COUNTER(dev);

    /*
     * 创建 ptimer: count
     * 使用 QEMU_CLOCK_VIRTUAL - 在 icount 模式下用 instruction counter
     *                          在普通模式下用 host timer
     */
    s->count_ptimer = ptimer_init(simple_counter_timer_tick, s,
                                                PTIMER_POLICY_DEFAULT);
    ptimer_transaction_begin(s->count_ptimer);
    /* 设置频率 (每秒多少次) - 根据 HDL 时钟频率调整 */
    ptimer_set_freq(s->count_ptimer, 1000000);  /* 1MHz */
    ptimer_transaction_commit(s->count_ptimer);

}

static void simple_counter_reset(DeviceState *dev)
{
    simple_counter_state *s = SIMPLE_COUNTER(dev);

    /* 重置 ptimer: count */
    ptimer_transaction_begin(s->count_ptimer);
    ptimer_stop(s->count_ptimer);
    ptimer_set_count(s->count_ptimer, 0);
    ptimer_transaction_commit(s->count_ptimer);
    s->count_limit = 0;
    qemu_irq_lower(s->irq);
}

static void simple_counter_class_init(ObjectClass *oc, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(oc);
    dc->realize = simple_counter_realize;
    dc->reset = simple_counter_reset;
}

static const TypeInfo simple_counter_info = {
    .name = TYPE_SIMPLE_COUNTER,
    .parent = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(simple_counter_state),
    .instance_init = simple_counter_init,
    .class_init = simple_counter_class_init,
};

static void simple_counter_register_types(void)
{
    type_register_static(&simple_counter_info);
}

type_init(simple_counter_register_types)
