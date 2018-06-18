import xlrd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import data_processing

data = data_processing.load_data(download=False)
new_data = data_processing.convert2onehot(data)


# prepare training data
new_data = new_data.values.astype(np.float32)       # change to numpy array and float32
np.random.shuffle(new_data)
sep = int(0.7*len(new_data))
train_data = new_data[:sep]                         # training data (70%)
test_data = new_data[sep:]                          # test data (30%)


# build network
tf_input = tf.placeholder(tf.float32, [None, 25], "input")
tfx = tf_input[:, :21]
tfy = tf_input[:, 21:]

l1 = tf.layers.dense(tfx, 128, tf.nn.relu, name="l1")
l2 = tf.layers.dense(l1, 128, tf.nn.relu, name="l2")
out = tf.layers.dense(l2, 4, name="l3")
prediction = tf.nn.softmax(out, name="pred")

loss = tf.losses.softmax_cross_entropy(onehot_labels=tfy, logits=out)
accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(out, axis=1),)[1]
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = opt.minimize(loss)

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

# training
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
accuracies, steps = [], []
for t in range(4000):
    # training
    batch_index = np.random.randint(len(train_data), size=32)
    sess.run(train_op, {tf_input: train_data[batch_index]})

    if t % 50 == 0:
        # testing
        acc_, pred_, loss_ = sess.run([accuracy, prediction, loss], {tf_input: test_data})
        accuracies.append(acc_)
        steps.append(t)
        print("Step: %i" % t,"| Accurate: %.2f" % acc_,"| Loss: %.2f" % loss_,)

        # visualize testing
        ax1.cla()
        for c in range(4):
            bp = ax1.bar(c+0.1, height=sum((np.argmax(pred_, axis=1) == c)), width=0.2, color='red')
            bt = ax1.bar(c-0.1, height=sum((np.argmax(test_data[:, 21:], axis=1) == c)), width=0.2, color='blue')
        ax1.set_xticks(range(4), ["unaccepted","accepted", "good", "very good"])
        ax1.legend(handles=[bp, bt], labels=["prediction", "target"])
        ax1.set_ylim((0, 400))
        ax2.cla()
        ax2.plot(steps, accuracies, label="accuracy")
        ax2.set_ylim(ymax=1)
        ax2.set_ylabel("accuracy")
        plt.pause(0.01)

plt.ioff()
plt.show()

# 获取excelfile对象
ExcelFile = xlrd.open_workbook(r'car.xlsx')
# 获取0号位置的表单
sheet = ExcelFile.sheet_by_index(0)

# 获取指定位置列的数据，并保存在数组中（标签）
buying_labels = sheet.col_values(8, 1, 5)
# 获取指定位置列的数据，并保存在数组中（数据）
buying_data = sheet.col_values(11, 1, 5)

# 打印标签与数据
print(buying_labels)
print(buying_data)

# 获取figure对象
fig = plt.figure()
# 画饼图（数据，数据对应的标签，百分数保留两位小数点）
plt.pie(buying_data, labels=buying_labels, autopct='%1.2f%%')
# 为图形命名
plt.title("buying")
# 保存图形
plt.savefig("buying.png")
# 显示图形
plt.show()

# 获取指定位置列的数据，并保存在数组中（标签）
maint_labels = sheet.col_values(8, 16, 20)
# 获取指定位置列的数据，并保存在数组中（数据）
maint_data = sheet.col_values(11, 16, 20)

# 打印标签与数据
print(maint_labels)
print(maint_data)

# 获取figure对象
fig = plt.figure()
# 画饼图（数据，数据对应的标签，百分数保留两位小数点）
plt.pie(maint_data, labels=maint_labels, autopct='%1.2f%%')
# 为图形命名
plt.title("maint")
 # 保存图形
plt.savefig("maint.png")
# 显示图形
plt.show()


# 获取指定位置列的数据，并保存在数组中（标签）
class_labels = sheet.col_values(8, 29, 33)
# 获取指定位置列的数据，并保存在数组中（数据）
class_data = sheet.col_values(11, 29, 33)

# 打印标签与数据
print(class_labels)
print(class_data)

# 获取figure对象
fig = plt.figure()
# 画饼图（数据，数据对应的标签，百分数保留两位小数点）
plt.pie(class_data, labels=class_labels, autopct='%1.2f%%')
# 为图形命名
plt.title("class")
# 保存图形
plt.savefig("class.png")
# 显示图形
plt.show()