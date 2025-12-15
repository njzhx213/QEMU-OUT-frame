module {
  hw.module @watchdog_timer(in %clk : i1, in %rst_n : i1, in %enable : i1, in %feed : i1, in %timeout_val : i32, out current_count : i32, out wdt_reset : i1, out wdt_warning : i1) {
    %0 = llhd.constant_time <0ns, 1d, 0e>
    %1 = llhd.constant_time <0ns, 0d, 1e>
    %c1_i32 = hw.constant 1 : i32
    %true = hw.constant true
    %false = hw.constant false
    %c0_i32 = hw.constant 0 : i32
    %clk_0 = llhd.sig name "clk" %false : i1
    %2 = llhd.prb %clk_0 : !hw.inout<i1>
    %rst_n_1 = llhd.sig name "rst_n" %false : i1
    %3 = llhd.prb %rst_n_1 : !hw.inout<i1>
    %enable_2 = llhd.sig name "enable" %false : i1
    %feed_3 = llhd.sig name "feed" %false : i1
    %timeout_val_4 = llhd.sig name "timeout_val" %c0_i32 : i32
    %wdt_reset = llhd.sig %false : i1
    %wdt_warning = llhd.sig %false : i1
    %counter = llhd.sig %c0_i32 : i32
    %warning_threshold = llhd.sig %c0_i32 : i32
    %4 = llhd.prb %counter : !hw.inout<i32>
    %5 = llhd.prb %timeout_val_4 : !hw.inout<i32>
    %6 = comb.extract %5 from 1 : (i32) -> i31
    %7 = comb.concat %false, %6 : i1, i31
    llhd.drv %warning_threshold, %7 after %1 : !hw.inout<i32>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 7 preds: ^bb0, ^bb2, ^bb4, ^bb7, ^bb9, ^bb10, ^bb11
      %10 = llhd.prb %clk_0 : !hw.inout<i1>
      %11 = llhd.prb %rst_n_1 : !hw.inout<i1>
      llhd.wait (%2, %3 : i1, i1), ^bb2
    ^bb2:  // pred: ^bb1
      %12 = llhd.prb %clk_0 : !hw.inout<i1>
      %13 = comb.xor bin %10, %true : i1
      %14 = comb.and bin %13, %12 : i1
      %15 = llhd.prb %rst_n_1 : !hw.inout<i1>
      %16 = comb.xor bin %15, %true : i1
      %17 = comb.and bin %11, %16 : i1
      %18 = comb.or bin %14, %17 : i1
      cf.cond_br %18, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %19 = llhd.prb %rst_n_1 : !hw.inout<i1>
      %20 = comb.xor %19, %true : i1
      cf.cond_br %20, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %counter, %c0_i32 after %0 : !hw.inout<i32>
      llhd.drv %wdt_reset, %false after %0 : !hw.inout<i1>
      llhd.drv %wdt_warning, %false after %0 : !hw.inout<i1>
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %21 = llhd.prb %enable_2 : !hw.inout<i1>
      cf.cond_br %21, ^bb6, ^bb11
    ^bb6:  // pred: ^bb5
      %22 = llhd.prb %feed_3 : !hw.inout<i1>
      cf.cond_br %22, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      llhd.drv %counter, %c0_i32 after %0 : !hw.inout<i32>
      llhd.drv %wdt_reset, %false after %0 : !hw.inout<i1>
      llhd.drv %wdt_warning, %false after %0 : !hw.inout<i1>
      cf.br ^bb1
    ^bb8:  // pred: ^bb6
      %23 = llhd.prb %counter : !hw.inout<i32>
      %24 = llhd.prb %timeout_val_4 : !hw.inout<i32>
      %25 = comb.icmp uge %23, %24 : i32
      cf.cond_br %25, ^bb9, ^bb10
    ^bb9:  // pred: ^bb8
      llhd.drv %wdt_reset, %true after %0 : !hw.inout<i1>
      cf.br ^bb1
    ^bb10:  // pred: ^bb8
      %26 = llhd.prb %counter : !hw.inout<i32>
      %27 = comb.add %26, %c1_i32 : i32
      llhd.drv %counter, %27 after %0 : !hw.inout<i32>
      %28 = llhd.prb %counter : !hw.inout<i32>
      %29 = llhd.prb %warning_threshold : !hw.inout<i32>
      %30 = comb.icmp uge %28, %29 : i32
      llhd.drv %wdt_warning, %30 after %0 : !hw.inout<i1>
      cf.br ^bb1
    ^bb11:  // pred: ^bb5
      llhd.drv %counter, %c0_i32 after %0 : !hw.inout<i32>
      llhd.drv %wdt_reset, %false after %0 : !hw.inout<i1>
      llhd.drv %wdt_warning, %false after %0 : !hw.inout<i1>
      cf.br ^bb1
    }
    llhd.drv %clk_0, %clk after %1 : !hw.inout<i1>
    llhd.drv %rst_n_1, %rst_n after %1 : !hw.inout<i1>
    llhd.drv %enable_2, %enable after %1 : !hw.inout<i1>
    llhd.drv %feed_3, %feed after %1 : !hw.inout<i1>
    llhd.drv %timeout_val_4, %timeout_val after %1 : !hw.inout<i32>
    %8 = llhd.prb %wdt_reset : !hw.inout<i1>
    %9 = llhd.prb %wdt_warning : !hw.inout<i1>
    hw.output %4, %8, %9 : i32, i1, i1
  }
}
