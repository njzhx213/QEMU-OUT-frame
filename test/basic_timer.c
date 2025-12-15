/*
 * Auto-generated QEMU device: basic_timer
 */

#include "qemu/osdep.h"
#include "hw/irq.h"
#include "hw/qdev-properties.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "sysemu/cpu-timers.h"  /* for icount */
#include "basic_timer.h"

/*
 * 读取 counter 计数器
 * 使用 icount 计算当前值，不需要每周期更新
 */
static uint32_t basic_timer_get_counter(basic_timer_state *s)
{
    if (!s->counter_running) {
        return s->counter_base;
    }

    /* 获取当前 icount */
    int64_t current_icount = icount_get();
    int64_t delta = current_icount - s->counter_icount_base;

    /* 根据方向计算当前值 */
    uint32_t current = s->counter_base + (delta * -1);
    return current;
}

/*
 * 写入 counter 计数器（加载新值）
 */
static void basic_timer_set_counter(basic_timer_state *s, uint32_t value)
{
    s->counter_base = value;
    s->counter_icount_base = icount_get();
}

static void basic_timer_start_counter(basic_timer_state *s)
{
    s->counter_icount_base = icount_get();
    s->counter_running = true;
}

static void basic_timer_stop_counter(basic_timer_state *s)
{
    /* 保存当前值 */
    s->counter_base = basic_timer_get_counter(s);
    s->counter_running = false;
}

static uint64_t basic_timer_read(void *opaque, hwaddr addr, unsigned size)
{
    basic_timer_state *s = opaque;
    uint64_t value = 0;

    switch (addr) {
    case 0x00:  /* running */
        value = s->running;
        break;
    case 0x04:  /* timeout_irq */
        value = s->timeout_irq;
        break;
    case 0x08:  /* counter */
        value = basic_timer_get_counter(s);
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR, "basic_timer: bad read at 0x%" HWADDR_PRIx "\n", addr);
    }
    return value;
}

static void basic_timer_write(void *opaque, hwaddr addr,
                              uint64_t value, unsigned size)
{
    basic_timer_state *s = opaque;

    switch (addr) {
    case 0x00:  /* running */
        s->running = value;
        break;
    case 0x04:  /* timeout_irq */
        s->timeout_irq = value;
        break;
    case 0x08:  /* counter */
        basic_timer_set_counter(s, value);
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR, "basic_timer: bad write at 0x%" HWADDR_PRIx "\n", addr);
    }
}

static const MemoryRegionOps basic_timer_ops = {
    .read = basic_timer_read,
    .write = basic_timer_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
};

static void basic_timer_init(Object *obj)
{
    basic_timer_state *s = BASIC_TIMER(obj);
    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);

    memory_region_init_io(&s->iomem, obj, &basic_timer_ops, s,
                          TYPE_BASIC_TIMER, 0x1000);
    sysbus_init_mmio(sbd, &s->iomem);
}

static void basic_timer_reset(DeviceState *dev)
{
    basic_timer_state *s = BASIC_TIMER(dev);

    s->running = 0;
    s->timeout_irq = 0;
    s->counter_base = 0;
    s->counter_icount_base = 0;
    s->counter_running = false;
}

static void basic_timer_class_init(ObjectClass *oc, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(oc);
    dc->reset = basic_timer_reset;
}

static const TypeInfo basic_timer_info = {
    .name = TYPE_BASIC_TIMER,
    .parent = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(basic_timer_state),
    .instance_init = basic_timer_init,
    .class_init = basic_timer_class_init,
};

static void basic_timer_register_types(void)
{
    type_register_static(&basic_timer_info);
}

type_init(basic_timer_register_types)
