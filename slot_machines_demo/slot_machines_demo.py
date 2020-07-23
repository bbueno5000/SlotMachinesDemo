"""
DOCSTRING
"""
import numpy
import tensorflow

tensorflow.compat.v1.disable_eager_execution()

class OneArmedBandits:
    """
    DOCSTRING
    """
    def __init__(self):
        self.bandits = [0.2, 0.0, -0.2, -5.0]
        self.num_bandits = len(self.bandits)
    
    def __call__(self):
        tensorflow.compat.v1.reset_default_graph()
        weights = tensorflow.Variable(tensorflow.ones([self.num_bandits]))
        chosen_action = tensorflow.argmax(weights, 0)
        reward_holder = tensorflow.keras.backend.placeholder(shape=[1], dtype=tensorflow.float32)
        action_holder = tensorflow.keras.backend.placeholder(shape=[1], dtype=tensorflow.int32)
        responsible_weight = tensorflow.slice(weights, action_holder, [1])
        optimizer = tensorflow.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
        loss = -(tensorflow.math.log(responsible_weight) * reward_holder)
        update = optimizer.minimize(loss)
        total_episodes = 1000
        total_reward = numpy.zeros(self.num_bandits)
        init = tensorflow.compat.v1.global_variables_initializer()
        with tensorflow.compat.v1.Session() as session:
            session.run(init)
            for i in range(total_episodes):
                if numpy.random.rand(1) < 0.1:
                    action = numpy.random.randint(self.num_bandits)
                else:
                    action = session.run(chosen_action)
                reward = self.pull_bandit(self.bandits[action])
                _, resp, ww = session.run(
                    [update, responsible_weight, weights],
                    feed_dict={reward_holder:[reward], action_holder:[action]})
                total_reward[action] += reward
                if i % 50 == 0:
                    print('Running reward:', str(total_reward))
        print('The agent selected bandit {} as the most promising.'.format(
            str(numpy.argmax(ww)+1)))
        if numpy.argmax(ww) == numpy.argmax(-numpy.array(self.bandits)):
            print('The agent was correct.')
        else:
            print('The agent was incorrect.')

    def pull_bandit(self, bandit):
        """
        DOCSTRING
        """
        result = numpy.random.randn(1)
        if result > bandit:
            return 1
        else:
            return -1

if __name__ == '__main__':
    one_armed_bandits = OneArmedBandits()
    one_armed_bandits()

