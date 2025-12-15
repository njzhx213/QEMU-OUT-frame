module {
  hw.module @simple_counter(in %clk : i1, in %rst_n : i1, out count : i8) {
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %1 = llhd.constant_time <0ns, 1d, 0e>
    %true = hw.constant true
    %c1_i8 = hw.constant 1 : i8
    %c0_i8 = hw.constant 0 : i8
    %false = hw.constant false
    %clk_0 = llhd.sig name "clk" %false : i1
    %2 = llhd.prb %clk_0 : !hw.inout<i1>
    %rst_n_1 = llhd.sig name "rst_n" %false : i1
    %3 = llhd.prb %rst_n_1 : !hw.inout<i1>
    %count = llhd.sig %c0_i8 : i8
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
      %5 = llhd.prb %clk_0 : !hw.inout<i1>
      %6 = llhd.prb %rst_n_1 : !hw.inout<i1>
      llhd.wait (%2, %3 : i1, i1), ^bb2
    ^bb2:  // pred: ^bb1
      %7 = llhd.prb %clk_0 : !hw.inout<i1>
      %8 = comb.xor bin %5, %true : i1
      %9 = comb.and bin %8, %7 : i1
      %10 = llhd.prb %rst_n_1 : !hw.inout<i1>
      %11 = comb.xor bin %10, %true : i1
      %12 = comb.and bin %6, %11 : i1
      %13 = comb.or bin %9, %12 : i1
      cf.cond_br %13, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %14 = llhd.prb %rst_n_1 : !hw.inout<i1>
      %15 = comb.xor %14, %true : i1
      cf.cond_br %15, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %count, %c0_i8 after %1 : !hw.inout<i8>
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %16 = llhd.prb %count : !hw.inout<i8>
      %17 = comb.add %16, %c1_i8 : i8
      llhd.drv %count, %17 after %1 : !hw.inout<i8>
      cf.br ^bb1
    }
    llhd.drv %clk_0, %clk after %0 : !hw.inout<i1>
    llhd.drv %rst_n_1, %rst_n after %0 : !hw.inout<i1>
    %4 = llhd.prb %count : !hw.inout<i8>
    hw.output %4 : i8
  }
}
