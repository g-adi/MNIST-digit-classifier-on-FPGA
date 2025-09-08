// sim/tb_top.v
`timescale 1ns/1ps

module tb_top;
  reg clk = 0;
  reg rst_n = 0;

  wire done;
  wire [3:0] predicted_digit;

  top dut(
    .clk(clk),
    .rst_n(rst_n),
    .done(done),
    .predicted_digit(predicted_digit)
  );

  // 100 MHz clock
  always #5 clk = ~clk;

  initial begin
    // waveform (for Icarus/GTKWave)
    $dumpfile("wave.vcd");
    $dumpvars(0, tb_top);

    // reset
    #50 rst_n = 1;

    // run until done fires a couple of times
    begin
      integer i;
      for (i = 0; i < 3; i = i + 1) begin
        @(posedge done);
        $display("[TB] Predicted digit = %0d (time=%0t)", predicted_digit, $time);
        #50;
      end
    end

    $finish;
  end

  // Point to your artifacts directory at runtime:
  // vvp sim.vvp +MEM_INIT_DIR=../hdl/mem_init
  // or pass via Verilator as +MEM_INIT_DIR=...
endmodule
