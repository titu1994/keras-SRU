# Keras Simple Recurrent Unit (SRU)
Initial implementation of Simple Recurrent Unit in Keras. Paper - [Training RNNs as Fast as CNNs](https://arxiv.org/abs/1709.02755)

**Note : Does not work perfectly yet**

# Issues

-[ ] Currently, no way to access the step counter (for each timestep) inside the K.rnn(...) loop. Therefore, cannot access the ith data point in the loop over time.

This is overcome by unrolling the RNN. Once unrolled, an internal python variable can be used to index the data point instead. Still considering on how to get the time variable inside the RNN loop.

-[ ] Input dim must exactly match the number of LSTM cells for now. Still working out how to overcome this problem.

-[ ] Performance of a single SRU layer is slightly lower (about 0.5% on average over 5 runs) compared to 1 layer LSTM (at least on IMDB, with batch size of 32). Haven't tried staking them yet, but this may improve performance.

Performance degrades substantially with larger batch sizes (about 6-7% on average over 5 runs) compared to 1 layer LSTM with batch size of 128.

However, a multi layer SRU (I've tried with 3 layers), while a bit slower than a 1 layer LSTM, gets around the same score on batch size of 32 or 128.

- Speed gains aren't that impressive at small batch size. At batch size of 32, SRU takes around 32-34 seconds. LSTM takes around 60-70 seconds. Thats just 50% reduction in speed, not the 5-10x that was discussed in the paper.

However, once batch size is increased to 128, SRU takes just 7 seconds per epoch compared to LSTM 22 seconds. For comparison, CNNs take 3-4 seconds per epoch.

