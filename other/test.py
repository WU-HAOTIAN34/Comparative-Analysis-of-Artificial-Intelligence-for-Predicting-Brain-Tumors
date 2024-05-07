import xlwt


string = '''
  6/264 [..............................] - ETA: 15s - loss: 1.0859 - accuracy: 0.4028WARNING:tensorflow:Callback method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0406s vs `on_train_batch_end` time: 0.0755s). Check your callbacks.
264/264 [==============================] - ETA: 0s - loss: 0.5767 - accuracy: 0.7557
Epoch 1: val_accuracy improved from -inf to 0.33207, saving model to CSDACNN2.h5
Epoch 1: 33.40 seconds
264/264 [==============================] - 33s 85ms/step - loss: 0.5767 - accuracy: 0.7557 - val_loss: 3.0441 - val_accuracy: 0.3321 - lr: 1.0000e-04
Epoch 2/50
264/264 [==============================] - ETA: 0s - loss: 0.3082 - accuracy: 0.8813
Epoch 2: val_accuracy improved from 0.33207 to 0.36053, saving model to CSDACNN2.h5
Epoch 2: 19.96 seconds
264/264 [==============================] - 20s 76ms/step - loss: 0.3082 - accuracy: 0.8813 - val_loss: 6.4181 - val_accuracy: 0.3605 - lr: 1.0000e-04
Epoch 3/50
264/264 [==============================] - ETA: 0s - loss: 0.2203 - accuracy: 0.9167
Epoch 3: val_accuracy improved from 0.36053 to 0.55313, saving model to CSDACNN2.h5
Epoch 3: 19.92 seconds
264/264 [==============================] - 20s 75ms/step - loss: 0.2203 - accuracy: 0.9167 - val_loss: 3.9704 - val_accuracy: 0.5531 - lr: 1.0000e-04
Epoch 4/50
264/264 [==============================] - ETA: 0s - loss: 0.1650 - accuracy: 0.9376
Epoch 4: val_accuracy did not improve from 0.55313
Epoch 4: 18.79 seconds
264/264 [==============================] - 19s 71ms/step - loss: 0.1650 - accuracy: 0.9376 - val_loss: 4.2223 - val_accuracy: 0.4744 - lr: 1.0000e-04
Epoch 5/50
263/264 [============================>.] - ETA: 0s - loss: 0.1252 - accuracy: 0.9552
Epoch 5: val_accuracy did not improve from 0.55313
Epoch 5: 19.04 seconds
264/264 [==============================] - 19s 72ms/step - loss: 0.1253 - accuracy: 0.9552 - val_loss: 2.9183 - val_accuracy: 0.5057 - lr: 1.0000e-04
Epoch 6/50
264/264 [==============================] - ETA: 0s - loss: 0.0933 - accuracy: 0.9680
Epoch 6: val_accuracy improved from 0.55313 to 0.92884, saving model to CSDACNN2.h5
Epoch 6: 19.73 seconds
264/264 [==============================] - 20s 75ms/step - loss: 0.0933 - accuracy: 0.9680 - val_loss: 0.2060 - val_accuracy: 0.9288 - lr: 1.0000e-04
Epoch 7/50
263/264 [============================>.] - ETA: 0s - loss: 0.0717 - accuracy: 0.9752
Epoch 7: val_accuracy did not improve from 0.92884
Epoch 7: 19.12 seconds
264/264 [==============================] - 19s 72ms/step - loss: 0.0718 - accuracy: 0.9750 - val_loss: 1.7815 - val_accuracy: 0.6480 - lr: 1.0000e-04
Epoch 8/50
264/264 [==============================] - ETA: 0s - loss: 0.0718 - accuracy: 0.9766
Epoch 8: val_accuracy did not improve from 0.92884
Epoch 8: 19.28 seconds
264/264 [==============================] - 19s 73ms/step - loss: 0.0718 - accuracy: 0.9766 - val_loss: 3.1197 - val_accuracy: 0.5987 - lr: 1.0000e-04
Epoch 9/50
264/264 [==============================] - ETA: 0s - loss: 0.0592 - accuracy: 0.9796
Epoch 9: val_accuracy did not improve from 0.92884
Epoch 9: 19.34 seconds
264/264 [==============================] - 19s 73ms/step - loss: 0.0592 - accuracy: 0.9796 - val_loss: 0.5108 - val_accuracy: 0.8776 - lr: 1.0000e-04
Epoch 10/50
263/264 [============================>.] - ETA: 0s - loss: 0.0557 - accuracy: 0.9801
Epoch 10: val_accuracy did not improve from 0.92884
Epoch 10: 18.17 seconds
264/264 [==============================] - 18s 69ms/step - loss: 0.0557 - accuracy: 0.9802 - val_loss: 2.3170 - val_accuracy: 0.6148 - lr: 1.0000e-04
Epoch 11/50
263/264 [============================>.] - ETA: 0s - loss: 0.0397 - accuracy: 0.9869
Epoch 11: val_accuracy did not improve from 0.92884

Epoch 11: ReduceLROnPlateau reducing learning rate to 2.9999999242136255e-05.
Epoch 11: 18.83 seconds
264/264 [==============================] - 19s 71ms/step - loss: 0.0399 - accuracy: 0.9868 - val_loss: 3.6865 - val_accuracy: 0.4583 - lr: 1.0000e-04
Epoch 12/50
263/264 [============================>.] - ETA: 0s - loss: 0.0131 - accuracy: 0.9960
Epoch 12: val_accuracy improved from 0.92884 to 0.97249, saving model to CSDACNN2.h5
Epoch 12: 19.27 seconds
264/264 [==============================] - 19s 73ms/step - loss: 0.0131 - accuracy: 0.9960 - val_loss: 0.0871 - val_accuracy: 0.9725 - lr: 3.0000e-05
Epoch 13/50
264/264 [==============================] - ETA: 0s - loss: 0.0078 - accuracy: 0.9982
Epoch 13: val_accuracy did not improve from 0.97249
Epoch 13: 19.39 seconds
264/264 [==============================] - 19s 73ms/step - loss: 0.0078 - accuracy: 0.9982 - val_loss: 0.3900 - val_accuracy: 0.8966 - lr: 3.0000e-05
Epoch 14/50
264/264 [==============================] - ETA: 0s - loss: 0.0056 - accuracy: 0.9991
Epoch 14: val_accuracy improved from 0.97249 to 0.97343, saving model to CSDACNN2.h5
Epoch 14: 19.87 seconds
264/264 [==============================] - 20s 75ms/step - loss: 0.0056 - accuracy: 0.9991 - val_loss: 0.0851 - val_accuracy: 0.9734 - lr: 3.0000e-05
Epoch 15/50
264/264 [==============================] - ETA: 0s - loss: 0.0037 - accuracy: 0.9996
Epoch 15: val_accuracy improved from 0.97343 to 0.98008, saving model to CSDACNN2.h5
Epoch 15: 19.82 seconds
264/264 [==============================] - 20s 75ms/step - loss: 0.0037 - accuracy: 0.9996 - val_loss: 0.0758 - val_accuracy: 0.9801 - lr: 3.0000e-05
Epoch 16/50
264/264 [==============================] - ETA: 0s - loss: 0.0062 - accuracy: 0.9982
Epoch 16: val_accuracy did not improve from 0.98008
Epoch 16: 19.34 seconds
264/264 [==============================] - 19s 73ms/step - loss: 0.0062 - accuracy: 0.9982 - val_loss: 0.1299 - val_accuracy: 0.9630 - lr: 3.0000e-05
Epoch 17/50
264/264 [==============================] - ETA: 0s - loss: 0.0107 - accuracy: 0.9964
Epoch 17: val_accuracy did not improve from 0.98008
Epoch 17: 19.38 seconds
264/264 [==============================] - 19s 73ms/step - loss: 0.0107 - accuracy: 0.9964 - val_loss: 0.7761 - val_accuracy: 0.8311 - lr: 3.0000e-05
Epoch 18/50
264/264 [==============================] - ETA: 0s - loss: 0.0084 - accuracy: 0.9978
Epoch 18: val_accuracy did not improve from 0.98008
Epoch 18: 19.04 seconds
264/264 [==============================] - 19s 72ms/step - loss: 0.0084 - accuracy: 0.9978 - val_loss: 0.8335 - val_accuracy: 0.7970 - lr: 3.0000e-05
Epoch 19/50
263/264 [============================>.] - ETA: 0s - loss: 0.0037 - accuracy: 0.9996
Epoch 19: val_accuracy did not improve from 0.98008
Epoch 19: 18.72 seconds
264/264 [==============================] - 19s 71ms/step - loss: 0.0037 - accuracy: 0.9996 - val_loss: 0.0788 - val_accuracy: 0.9725 - lr: 3.0000e-05
Epoch 20/50
264/264 [==============================] - ETA: 0s - loss: 0.0054 - accuracy: 0.9984
Epoch 20: val_accuracy did not improve from 0.98008

Epoch 20: ReduceLROnPlateau reducing learning rate to 8.999999772640877e-06.
Epoch 20: 19.12 seconds
264/264 [==============================] - 19s 72ms/step - loss: 0.0054 - accuracy: 0.9984 - val_loss: 0.2608 - val_accuracy: 0.9440 - lr: 3.0000e-05
Epoch 21/50
264/264 [==============================] - ETA: 0s - loss: 0.0032 - accuracy: 0.9992
Epoch 21: val_accuracy did not improve from 0.98008
Epoch 21: 18.51 seconds
264/264 [==============================] - 19s 70ms/step - loss: 0.0032 - accuracy: 0.9992 - val_loss: 0.0632 - val_accuracy: 0.9801 - lr: 9.0000e-06
Epoch 22/50
263/264 [============================>.] - ETA: 0s - loss: 0.0015 - accuracy: 0.9999
Epoch 22: val_accuracy improved from 0.98008 to 0.98292, saving model to CSDACNN2.h5
Epoch 22: 20.16 seconds
264/264 [==============================] - 20s 76ms/step - loss: 0.0015 - accuracy: 0.9999 - val_loss: 0.0611 - val_accuracy: 0.9829 - lr: 9.0000e-06
Epoch 23/50
264/264 [==============================] - ETA: 0s - loss: 0.0018 - accuracy: 0.9998
Epoch 23: val_accuracy did not improve from 0.98292
Epoch 23: 19.18 seconds
264/264 [==============================] - 19s 73ms/step - loss: 0.0018 - accuracy: 0.9998 - val_loss: 0.0537 - val_accuracy: 0.9829 - lr: 9.0000e-06
Epoch 24/50
264/264 [==============================] - ETA: 0s - loss: 0.0014 - accuracy: 1.0000
Epoch 24: val_accuracy improved from 0.98292 to 0.98861, saving model to CSDACNN2.h5
Epoch 24: 19.49 seconds
264/264 [==============================] - 19s 74ms/step - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.0482 - val_accuracy: 0.9886 - lr: 9.0000e-06
Epoch 25/50
264/264 [==============================] - ETA: 0s - loss: 0.0016 - accuracy: 0.9997
Epoch 25: val_accuracy did not improve from 0.98861
Epoch 25: 18.73 seconds
264/264 [==============================] - 19s 71ms/step - loss: 0.0016 - accuracy: 0.9997 - val_loss: 0.0638 - val_accuracy: 0.9848 - lr: 9.0000e-06
Epoch 26/50
263/264 [============================>.] - ETA: 0s - loss: 0.0017 - accuracy: 0.9999
Epoch 26: val_accuracy did not improve from 0.98861
Epoch 26: 19.09 seconds
264/264 [==============================] - 19s 72ms/step - loss: 0.0021 - accuracy: 0.9998 - val_loss: 0.0650 - val_accuracy: 0.9810 - lr: 9.0000e-06
Epoch 27/50
264/264 [==============================] - ETA: 0s - loss: 0.0023 - accuracy: 0.9997
Epoch 27: val_accuracy did not improve from 0.98861
Epoch 27: 18.90 seconds
264/264 [==============================] - 19s 72ms/step - loss: 0.0023 - accuracy: 0.9997 - val_loss: 0.0679 - val_accuracy: 0.9810 - lr: 9.0000e-06
Epoch 28/50
264/264 [==============================] - ETA: 0s - loss: 0.0011 - accuracy: 0.9999
Epoch 28: val_accuracy did not improve from 0.98861
Epoch 28: 18.81 seconds
264/264 [==============================] - 19s 71ms/step - loss: 0.0011 - accuracy: 0.9999 - val_loss: 0.0651 - val_accuracy: 0.9820 - lr: 9.0000e-06
Epoch 29/50
264/264 [==============================] - ETA: 0s - loss: 0.0020 - accuracy: 0.9993
Epoch 29: val_accuracy did not improve from 0.98861

Epoch 29: ReduceLROnPlateau reducing learning rate to 2.6999998226528985e-06.
Epoch 29: 18.86 seconds
264/264 [==============================] - 19s 71ms/step - loss: 0.0020 - accuracy: 0.9993 - val_loss: 0.1241 - val_accuracy: 0.9696 - lr: 9.0000e-06
Epoch 30/50
264/264 [==============================] - ETA: 0s - loss: 8.3848e-04 - accuracy: 0.9999
Epoch 30: val_accuracy did not improve from 0.98861
Epoch 30: 18.92 seconds
264/264 [==============================] - 19s 72ms/step - loss: 8.3848e-04 - accuracy: 0.9999 - val_loss: 0.0584 - val_accuracy: 0.9829 - lr: 2.7000e-06
Epoch 31/50
264/264 [==============================] - ETA: 0s - loss: 0.0011 - accuracy: 0.9999
Epoch 31: val_accuracy did not improve from 0.98861
Epoch 31: 17.28 seconds
264/264 [==============================] - 17s 65ms/step - loss: 0.0011 - accuracy: 0.9999 - val_loss: 0.0675 - val_accuracy: 0.9810 - lr: 2.7000e-06
Epoch 32/50
264/264 [==============================] - ETA: 0s - loss: 0.0012 - accuracy: 0.9998
Epoch 32: val_accuracy did not improve from 0.98861
Epoch 32: 17.75 seconds
264/264 [==============================] - 18s 67ms/step - loss: 0.0012 - accuracy: 0.9998 - val_loss: 0.0593 - val_accuracy: 0.9810 - lr: 2.7000e-06
Epoch 33/50
263/264 [============================>.] - ETA: 0s - loss: 6.9428e-04 - accuracy: 1.0000
Epoch 33: val_accuracy did not improve from 0.98861
Epoch 33: 18.77 seconds
264/264 [==============================] - 19s 71ms/step - loss: 6.9807e-04 - accuracy: 1.0000 - val_loss: 0.0547 - val_accuracy: 0.9820 - lr: 2.7000e-06
Epoch 34/50
264/264 [==============================] - ETA: 0s - loss: 0.0010 - accuracy: 0.9999    
Epoch 34: val_accuracy did not improve from 0.98861

Epoch 34: ReduceLROnPlateau reducing learning rate to 8.099999604382901e-07.
Epoch 34: 18.37 seconds
264/264 [==============================] - 18s 70ms/step - loss: 0.0010 - accuracy: 0.9999 - val_loss: 0.0576 - val_accuracy: 0.9829 - lr: 2.7000e-06
Epoch 35/50
264/264 [==============================] - ETA: 0s - loss: 5.7821e-04 - accuracy: 1.0000
Epoch 35: val_accuracy did not improve from 0.98861
Epoch 35: 18.25 seconds
264/264 [==============================] - 18s 69ms/step - loss: 5.7821e-04 - accuracy: 1.0000 - val_loss: 0.0571 - val_accuracy: 0.9820 - lr: 8.1000e-07
Epoch 36/50
264/264 [==============================] - ETA: 0s - loss: 5.3627e-04 - accuracy: 1.0000
Epoch 36: val_accuracy did not improve from 0.98861
Epoch 36: 19.42 seconds
264/264 [==============================] - 19s 74ms/step - loss: 5.3627e-04 - accuracy: 1.0000 - val_loss: 0.0568 - val_accuracy: 0.9810 - lr: 8.1000e-07
Epoch 37/50
264/264 [==============================] - ETA: 0s - loss: 6.5073e-04 - accuracy: 1.0000
Epoch 37: val_accuracy did not improve from 0.98861
Epoch 37: 19.04 seconds
264/264 [==============================] - 19s 72ms/step - loss: 6.5073e-04 - accuracy: 1.0000 - val_loss: 0.0564 - val_accuracy: 0.9820 - lr: 8.1000e-07
Epoch 38/50
264/264 [==============================] - ETA: 0s - loss: 5.1296e-04 - accuracy: 1.0000
Epoch 38: val_accuracy did not improve from 0.98861
Epoch 38: 19.28 seconds
264/264 [==============================] - 19s 73ms/step - loss: 5.1296e-04 - accuracy: 1.0000 - val_loss: 0.0566 - val_accuracy: 0.9801 - lr: 8.1000e-07
Epoch 39/50
264/264 [==============================] - ETA: 0s - loss: 7.3745e-04 - accuracy: 0.9998
Epoch 39: val_accuracy did not improve from 0.98861

Epoch 39: ReduceLROnPlateau reducing learning rate to 2.4299998813148704e-07.
Epoch 39: 18.13 seconds
264/264 [==============================] - 18s 69ms/step - loss: 7.3745e-04 - accuracy: 0.9998 - val_loss: 0.0559 - val_accuracy: 0.9839 - lr: 8.1000e-07
Epoch 40/50
263/264 [============================>.] - ETA: 0s - loss: 9.6958e-04 - accuracy: 0.9998
Epoch 40: val_accuracy did not improve from 0.98861
Epoch 40: 19.42 seconds
264/264 [==============================] - 19s 74ms/step - loss: 9.7043e-04 - accuracy: 0.9998 - val_loss: 0.0539 - val_accuracy: 0.9829 - lr: 2.4300e-07
Epoch 41/50
264/264 [==============================] - ETA: 0s - loss: 7.9289e-04 - accuracy: 0.9999
Epoch 41: val_accuracy did not improve from 0.98861
Epoch 41: 19.16 seconds
264/264 [==============================] - 19s 73ms/step - loss: 7.9289e-04 - accuracy: 0.9999 - val_loss: 0.0526 - val_accuracy: 0.9839 - lr: 2.4300e-07
Epoch 42/50
264/264 [==============================] - ETA: 0s - loss: 0.0012 - accuracy: 0.9997
Epoch 42: val_accuracy did not improve from 0.98861
Epoch 42: 19.43 seconds
264/264 [==============================] - 19s 74ms/step - loss: 0.0012 - accuracy: 0.9997 - val_loss: 0.0546 - val_accuracy: 0.9848 - lr: 2.4300e-07
Epoch 43/50
263/264 [============================>.] - ETA: 0s - loss: 4.4472e-04 - accuracy: 1.0000
Epoch 43: val_accuracy did not improve from 0.98861
Epoch 43: 19.09 seconds
264/264 [==============================] - 19s 72ms/step - loss: 4.4470e-04 - accuracy: 1.0000 - val_loss: 0.0518 - val_accuracy: 0.9839 - lr: 2.4300e-07
Epoch 44/50
263/264 [============================>.] - ETA: 0s - loss: 7.0193e-04 - accuracy: 0.9999
Epoch 44: val_accuracy did not improve from 0.98861

Epoch 44: ReduceLROnPlateau reducing learning rate to 7.289999643944612e-08.
Epoch 44: 18.63 seconds
264/264 [==============================] - 19s 71ms/step - loss: 7.0123e-04 - accuracy: 0.9999 - val_loss: 0.0516 - val_accuracy: 0.9839 - lr: 2.4300e-07
Epoch 45/50
264/264 [==============================] - ETA: 0s - loss: 9.5955e-04 - accuracy: 0.9997
Epoch 45: val_accuracy did not improve from 0.98861
Epoch 45: 18.72 seconds
264/264 [==============================] - 19s 71ms/step - loss: 9.5955e-04 - accuracy: 0.9997 - val_loss: 0.0537 - val_accuracy: 0.9829 - lr: 7.2900e-08
Epoch 46/50
264/264 [==============================] - ETA: 0s - loss: 6.0649e-04 - accuracy: 1.0000
Epoch 46: val_accuracy did not improve from 0.98861
Epoch 46: 19.47 seconds
264/264 [==============================] - 19s 74ms/step - loss: 6.0649e-04 - accuracy: 1.0000 - val_loss: 0.0521 - val_accuracy: 0.9839 - lr: 7.2900e-08
Epoch 47/50
264/264 [==============================] - ETA: 0s - loss: 6.2379e-04 - accuracy: 0.9999
Epoch 47: val_accuracy did not improve from 0.98861
Epoch 47: 19.23 seconds
264/264 [==============================] - 19s 73ms/step - loss: 6.2379e-04 - accuracy: 0.9999 - val_loss: 0.0524 - val_accuracy: 0.9839 - lr: 7.2900e-08
Epoch 48/50
264/264 [==============================] - ETA: 0s - loss: 7.8275e-04 - accuracy: 0.9998
Epoch 48: val_accuracy did not improve from 0.98861
Epoch 48: 19.04 seconds
264/264 [==============================] - 19s 72ms/step - loss: 7.8275e-04 - accuracy: 0.9998 - val_loss: 0.0521 - val_accuracy: 0.9829 - lr: 7.2900e-08
Epoch 49/50
264/264 [==============================] - ETA: 0s - loss: 7.3407e-04 - accuracy: 1.0000
Epoch 49: val_accuracy did not improve from 0.98861

Epoch 49: ReduceLROnPlateau reducing learning rate to 2.1869998079182552e-08.
Epoch 49: 19.81 seconds
264/264 [==============================] - 20s 75ms/step - loss: 7.3407e-04 - accuracy: 1.0000 - val_loss: 0.0528 - val_accuracy: 0.9829 - lr: 7.2900e-08
Epoch 50/50
264/264 [==============================] - ETA: 0s - loss: 4.8420e-04 - accuracy: 1.0000
Epoch 50: val_accuracy did not improve from 0.98861
Epoch 50: 19.35 seconds
264/264 [==============================] - 19s 73ms/step - loss: 4.8420e-04 - accuracy: 1.0000 - val_loss: 0.0520 - val_accuracy: 0.9829 - lr: 2.1870e-08
Total training time: 968.08 seconds

'''
temp = string.split('\n')
test = [item.split('264/264 [==============================]')[1] for item in temp if 'val_loss:' in item.split(' ')]
loss = []
accuracy = []
val_accuracy = []
val_loss = []
for item in test:
    temp = item.split('- ')
    temp.remove(' ')
    for num in temp:
        if num.split(' ')[0] == 'loss:':
            loss.append(num.split(' ')[1])
        if num.split(' ')[0] == 'accuracy:':
            accuracy.append(num.split(' ')[1])
        if num.split(' ')[0] == 'val_accuracy:':
            val_accuracy.append(num.split(' ')[1])
        if num.split(' ')[0] == 'val_loss:':
            val_loss.append(num.split(' ')[1])

print(val_loss)
print(val_accuracy)
print(accuracy)
print(loss)

f = xlwt.Workbook('encoding = utf-8')  # 设置工作簿编码
sheet1 = f.add_sheet('sheet1', cell_overwrite_ok=True)  # 创建sheet工作表
sheet1.write(0, 0, 'loss')
sheet1.write(0, 1, 'accuracy')
sheet1.write(0, 2, 'val_loss')
sheet1.write(0, 3, 'val_accuracy')

for i in range(len(loss)):
    sheet1.write(i+1, 0, loss[i])
    sheet1.write(i+1, 1, accuracy[i])
    sheet1.write(i+1, 2, val_loss[i])
    sheet1.write(i+1, 3, val_accuracy[i])
f.save('text15.xls')  # 保存.xls到当前工作目录

