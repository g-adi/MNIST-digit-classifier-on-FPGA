// hdl/fc2_layer.v
`timescale 1ns/1ps

// Serial fully-connected layer that outputs int32 logits (no shift, no ReLU).
// Reads:  h1_mem (int8), W2 (int8, row-major), b2 (int32)
// Writes: y2_mem via write port (int32)
module fc2_layer #(
  parameter integer IN_DIM  = 32,
  parameter integer OUT_DIM = 10
)(
  input  wire                 clk,
  input  wire                 rst,
  input  wire                 start,
  output reg                  done,

  // READ interfaces
  output reg  [$clog2(IN_DIM)-1:0]    x_addr,
  input  wire signed [7:0]            x_data,

  output reg  [$clog2(IN_DIM*OUT_DIM)-1:0] w_addr,
  input  wire signed [7:0]            w_data,

  output reg  [$clog2(OUT_DIM)-1:0]   b_addr,
  input  wire signed [31:0]           b_data,

  // WRITE port for logits (int32)
  output reg                          y_we,
  output reg  [$clog2(OUT_DIM)-1:0]   y_addr,
  output reg  signed [31:0]           y_data
);

  localparam S_IDLE      = 3'd0;
  localparam S_LOAD_BIAS = 3'd1;
  localparam S_MAC       = 3'd2;
  localparam S_WRITE     = 3'd3;
  localparam S_DONE      = 3'd4;

  reg [2:0] state;

  reg [$clog2(OUT_DIM)-1:0] out_counter;
  reg [$clog2(IN_DIM)-1:0]  in_counter;

  reg signed [31:0] acc;

  wire [$clog2(IN_DIM*OUT_DIM)-1:0] w_flat_addr = out_counter*IN_DIM + in_counter;

  always @(posedge clk) begin
    if (rst) begin
      state <= S_IDLE; done <= 1'b0;
      out_counter <= {$clog2(OUT_DIM){1'b0}}; in_counter <= {$clog2(IN_DIM){1'b0}};
      acc <= 32'sd0;
      x_addr <= {$clog2(IN_DIM){1'b0}}; w_addr <= {$clog2(IN_DIM*OUT_DIM){1'b0}}; b_addr <= {$clog2(OUT_DIM){1'b0}};
      y_we <= 1'b0; y_addr <= {$clog2(OUT_DIM){1'b0}}; y_data <= 32'sd0;
    end else begin
      y_we <= 1'b0;
      case (state)
        S_IDLE: begin
          done <= 1'b0;
          if (start) begin
            out_counter <= {$clog2(OUT_DIM){1'b0}};
            in_counter  <= {$clog2(IN_DIM){1'b0}};
            b_addr      <= {$clog2(OUT_DIM){1'b0}};
            state       <= S_LOAD_BIAS;
          end
        end

        S_LOAD_BIAS: begin
          b_addr <= out_counter;
          acc    <= b_data;            // init with bias
          in_counter <= {$clog2(IN_DIM){1'b0}};
          x_addr     <= {$clog2(IN_DIM){1'b0}};
          w_addr     <= w_flat_addr;
          state      <= S_MAC;
        end

        S_MAC: begin
          acc <= acc + $signed(w_data) * $signed(x_data);
          if (in_counter == IN_DIM-1) begin
            state <= S_WRITE;
          end else begin
            in_counter <= in_counter + 1'b1;
            x_addr     <= in_counter + 1'b1;
            w_addr     <= out_counter*IN_DIM + (in_counter + 1'b1);
          end
        end

        S_WRITE: begin
          y_addr <= out_counter;
          y_data <= acc;
          y_we   <= 1'b1;

          if (out_counter == OUT_DIM-1) begin
            state <= S_DONE;
          end else begin
            out_counter <= out_counter + 1'b1;
            state       <= S_LOAD_BIAS;
          end
        end

        S_DONE: begin
          done  <= 1'b1;
          state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
