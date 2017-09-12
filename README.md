# Keras 
Initial implementation of Simple Recurrent Unit in Keras.

**Note : Does not work perfectly yet **

# Issues

- Currently, no way to access the step counter (for each timestep) inside the K.rnn(...) loop. Therefore, cannot access the ith data point in the loop over time. 

This is overcome by unrolling the RNN. Once unrolled, an internal python variable can be used to index the data point instead. Still considering on how to get the time variable inside the RNN loop.

- Input dim must exactly match the number of LSTM cells for now. Still working out how to overcome this problem.

- Performance of a single SRU layer is lower compared to 1 layer LSTM. Haven't tried staking them yet.
