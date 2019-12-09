import tensorflow as tf

zero=tf.Variable(0)

one=tf.constant(1)

new_value=tf.add(zero,one)

update=tf.assign(zero,new_value)

print(update)

init_op=tf.global_variables_initializer()

sess=tf.Session()

sess.run(init_op)

print(sess.run(zero))


for _ in range(10):
    sess.run(update)
    print(sess.run(zero))

hello=tf.constant("hello")

world=tf.constant("world")

helloworld=tf.add(hello,world)

print(sess.run(helloworld))

## Placeholders

a=tf.placeholder(tf.float32)
b=a*2

result=sess.run(b,feed_dict={a:3})

print(result)



result=sess.run(b,feed_dict={a:[3,4,5]})

print(result)