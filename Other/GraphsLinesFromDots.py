import numpy as np
from matplotlib import pyplot as plt

acc = np.array([
0.2463,
0.5179,
0.6746,
0.7629,
0.8281,
0.8608,
0.8879,
0.9175,
0.9206,
0.9344,
0.9438,
0.9510,
0.9568,
0.9639,
0.9695,
0.9734,
0.9764,
0.9797,
0.9831,
0.9812,
0.9855,
0.9846,
0.9865,
0.9876,
0.9916,
0.9902,
0.9948,
0.9940,
0.9951,
0.9943])
val_acc = np.array([
0.3406,
0.4357,
0.5134,
0.5886,
0.6438,
0.6874,
0.7267,
0.7569,
0.7783,
0.7906,
0.8008,
0.8081,
0.8171,
0.8210,
0.8257,
0.8296,
0.8323,
0.8372,
0.8374,
0.8384,
0.8409,
0.8444,
0.8448,
0.8493,
0.8512,
0.8513,
0.8545,
0.8548,
0.8552,
0.8575,
])
loss = np.array([
2.4965,
1.3496,
0.9018,
0.6811,
0.5087,
0.4075,
0.3451,
0.2692,
0.2564,
0.2144,
0.1862,
0.1636,
0.1491,
0.1332,
0.1112,
0.1038,
0.0931,
0.0843,
0.0772,
0.0735,
0.0634,
0.0608,
0.0567,
0.0513,
0.0466,
0.0443,
0.0363,
0.0350,
0.0315,
0.0303])
val_loss = np.array([
1.9108,
1.5790,
1.3343,
1.1339,
0.9795,
0.8552,
0.7619,
0.6954,
0.6421,
0.6044,
0.5781,
0.5601,
0.5430,
0.5366,
0.5280,
0.5277,
0.5230,
0.5190,
0.5182,
0.5222,
0.5191,
0.5157,
0.5167,
0.5127,
0.5139,
0.5187,
0.5153,
0.5217,
0.5248,
0.5228])

plt.plot(np.arange(30), acc, 'b-', markersize=2, label='Dokładność treningowa')
plt.plot(np.arange(30), val_acc, 'g-', markersize=2, label='Dokładność walidacyjna')
plt.title('Dokładność i strata treningowa i walidacyjna')
# plt.legend()

# plt.figure()

plt.plot(np.arange(30), loss, 'c-', markersize=2, label='Strata treningowa')
plt.plot(np.arange(30), val_loss, 'r-', markersize=2, label='Strata walidacyjna')
# plt.title('Strata treningowa i walidacyjna')
plt.legend()

plt.show()