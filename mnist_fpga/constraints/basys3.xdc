# Basys3 Constraints for MNIST Neural Network
# Clock signal (100MHz)
set_property PACKAGE_PIN W5 [get_ports clk]							
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports clk]

# Reset button (center button)
set_property PACKAGE_PIN U18 [get_ports rst_n]						
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]

# Start button (left button)
set_property PACKAGE_PIN T18 [get_ports start_btn]						
set_property IOSTANDARD LVCMOS33 [get_ports start_btn]

# LEDs for predicted digit output (4 bits = 0-15, but we only use 0-9)
set_property PACKAGE_PIN U16 [get_ports {predicted_digit[0]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {predicted_digit[0]}]
set_property PACKAGE_PIN E19 [get_ports {predicted_digit[1]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {predicted_digit[1]}]
set_property PACKAGE_PIN U19 [get_ports {predicted_digit[2]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {predicted_digit[2]}]
set_property PACKAGE_PIN V19 [get_ports {predicted_digit[3]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {predicted_digit[3]}]

# Done signal LED
set_property PACKAGE_PIN W18 [get_ports done]					
set_property IOSTANDARD LVCMOS33 [get_ports done]

# Timing constraints for synthesis
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets clk_IBUF]
