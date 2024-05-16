import spidev
import time
import datetime
import csv
import joblib
import numpy as np

# --- ADC Configuration (Replace with your setup)---
spi = spidev.SpiDev(0, 0)
spi.max_speed_hz = 1000000

def read_adc(channel):
    # Construct a three-byte message for SPI communication
    # First byte (MSB)
    #   - Start bit (1)
    #   - Single-ended conversion (0)
    #   - Channel selection bits (2 bits for channel 0, 1, or 2)
    # Second and third bytes (ignored by MCP308)
    #   - Set to 0 for continuous conversion mode
    msg = [0x01, 0x80 | (channel & 0x07) << 4, 0]

    # Send the message and receive the response
    adc_data = spi.xfer2(msg)

    # Combine the second and third bytes to form a 10-bit digital value
    # (discard the first byte since it's the start bit and channel selection)
    digital_value = (adc_data[1] & 0x1F) << 8 | adc_data[2]

    return digital_value  

# --- Load your pre-trained models ---
linear_model = joblib.load('LR_OV_Sym_Inst.pkl')
rf_model = joblib.load('RF_OV_Sym_Inst.pkl')

# --- Timed Execution --- 
duration = 2 * 60  # Duration in seconds
filename = "adc_predictions.csv"

with open(filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Timestamp", "Va", "Vb", "Vc", "Ia", "Ib",  "Ic", "RUL"])

    while True:  # Main loop to ask for continuation
        start_time = datetime.datetime.now()

        while True:  # Inner loop for ADC reads within the time limit
            current_time = datetime.datetime.now()
            elapsed_time = (current_time - start_time).total_seconds()

            if elapsed_time >= duration:
                break  # Exit the inner loop if time limit reached

            # --- Collect ADC Data ---
            channel_0_value = read_adc(0)
            channel_1_value = read_adc(1)
            channel_2_value = read_adc(2)
            channel_3_value = read_adc(3)
            channel_4_value = read_adc(4)
            channel_5_value = read_adc(5)

            # --- Voltage Conversion ---
            x = ((channel_0_value) * 3.3) / 1023
            x = (((x - 1.24) / 1.24) * (130 * 1.414))

            x2 = ((channel_1_value) * 3.3) / 1023
            x2 = (((x2 - 1.24) / 1.24) * (130 * 1.414))

            x3 = ((channel_2_value) * 3.3) / 1023
            x3 = (((x3 - 1.27) / 1.27) * (130 * 1.414))
            
            # --- current --
            c = ((channel_3_value) * 3.3) / 1023
            c = (((c - 1.24) / 1.24) * (4 * 1.414))
            c2 = ((channel_4_value) * 3.3) / 1023
            c2 = (((c2 - 1.24) / 1.24) * (4 * 1.414))
            c3 = ((channel_5_value) * 3.3) / 1023
            c3 = (((c3 - 1.24) / 1.24) * (4 * 1.414))

            # --- Create Input Data ---
            input_data = np.array([x, x2, x3, c, c2, c3]).reshape(1, -1) 

            # --- Make Predictions ---
            predictions_lr = linear_model.predict(input_data)
            predictions_rf = rf_model.predict(input_data)

            # --- Calculate RUL ---
            Lo = 20000
            Tb = 155
            avg_temp = np.mean(predictions_lr)  
            Tc = avg_temp  # Using the average temperature
            RUL = (Lo * 2**((Tb - Tc) / 10)) / (365 * 24)  # RUL in years

            # --- Write to CSV ---
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            row = [timestamp, x, x2, x3, c, c2, c3, RUL]  # Added raw ADC values
            csv_writer.writerow(row) 

            # --- Utilize the Predictions --- (adjust as needed) ... 
            print("Linear Regression Prediction: Phase A Temp:", predictions_lr[0])  
            print("Random Forest Prediction:", predictions_rf[0])
            print("RUL (LR): ", RUL)

        # --- Ask to Continue ---
        user_input = input("Do you want to continue? (yes/no): ")
        if user_input.lower() != 'yes':
            print("Exiting script...")
            break  # Exit the outer loop
