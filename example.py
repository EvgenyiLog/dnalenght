import pandas as pd
import numpy as np
from readerfrf import parse_frf_file
import matplotlib.pyplot as plt
from subtract_reference_from_columns import subtract_reference_from_columns
from msbackadj import msbackadj
import pyyawt
def main():
    matrix_df, channels_df, metadata=parse_frf_file( r"C:\Users\Admin\Documents\GitHub\dnalenght\files\Anton_lib_test_2_17_56_29\0.1-5-0.2_F9.frf")
    print(channels_df.columns)
    plt.figure() 
    channels_df.loc[:,'dR110'].plot()
    plt.grid(True)


    df=subtract_reference_from_columns(channels_df,50)
    plt.figure() 
    df.loc[:,'dR110'].plot()
    plt.grid(True)

    time = np.arange(len(channels_df))  # номер отсчёта

    # исходный сигнал
    signal = df['dR110']

    # применяем msbackadj
    signal_corrected = msbackadj(time, signal.values)

    # добавляем в тот же DataFrame
    df['dR110_corr'] = signal_corrected
    plt.figure() 
    df.loc[:,'dR110'].plot()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
