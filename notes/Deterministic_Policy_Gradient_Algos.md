### Deterministic Policy Gradient Algorithms 


+ Purpose: An algorithm meant to address the problem of continuous actions

##### Abstract
+ The deterministic policy gradients appeal is that it is the expected graident of the action-value function. 
    - This simple form means the deterministic policy gradient can be estimated much more efficiently than the usual stochastic policy gradient. 
+ To ensure adequate exploration, an _off-policy actor-critic_ algorithm is introdced that learns a deterministic target policy from an exploratory behaviour policy. 

#### TODO = read the rest of the paper on DDPG and take notes of it HERE 

### How-To 

#### Theory 
+ *Policy-Gradient Methods* 
    - Policy-Gradient algorithms optimize a policy end-to-end by computing noisy estimates of the gradient of the expected reward of the policy and then updating the policy in the gradient direction. 
    - The issue with using a vanilla PG method on our experiment is that we would only reward at the end, and to give an agent only on signal after a long experience makes it difficult for the agent to ascertain exactly which action was good, this is known as the *credit assignment problem*.
    
+ *Actor-Critic Algorithms* 
    - The actor-critic learning algorithm is used to represent the policy function independently of the value function. 
    - The policy function is known as the *actor* 
    - The value function is known as the *critic*.  
        - The actor produces action given the current state of the environment, and the critic produces a TD (Temporal-Difference) error signal given the state and resultant reward. 
    - If the critic is estimating the action-value function $Q(s,a)$, it will also need the output of the actor. 
    - The output of the critic drives learning in both the actor and the critic. 
    - Neural Networks can be used to represent the actor and critic structures.
    
+ *Off-Policy vs. On-Policy*
    - Off-policy = employ a seperate behavior policy that is indepenedent of the policy being improved upon; the behavior policy is used to simulate trajectories. 
    - A key benefit = the behavior policy can operate by sampling all actions, whereas the estimation policy can be deterministic (e.g. greedy). 
    - Q-learning is an off-policy algorithm, since it updates the Q values without making any assumptions about the actual policy being followed. 
    - Rather, the Q-learning algorithm simply states that the Q-value corresponding to state $s(t)$ and action $a(t)$ is updated using the Q-value of the next state $s(t+1)$ and the action $a(t+1)$ that maximizes the Q-value at state $s(t + 1)$.

+ *Model-Free Algorithms*
    - Model-free RL algorithms make no effort to learn the underlying dynamics that govern how an agent interacts with the environment. 
      - THIS IS SUPER IMPORTANT!! WE CAN APPLY GENERIC ALGORITHM TO OUR EXPERIMENT WITH EASE.
    - Rather than dealing with high complexity that stochastic matrix (matrix representing actions and consequential states from a given state in a stochastic process) provides in a problem as complex as trajectory, model free algorithms estimate the optimal policy or value function through algorithms such as policy iteration or value iteration. 
    - Nonetheless, it still serves well to have a good model of the environment.
    - Model-free algorithms generally require a larger number of training examples. 

+ *Meat and Potatoes of DDPG*
    - At it's core, DDPG is a policy gradient algorithm that uses a stochastic behavior policy for good exploration but estimates a deterministic target policy, which is much easier to learn.
    - Policy gradient algorithms utilize a form of policy iteration: they evaluate the policy, and then follow the policy gradient to maximize performance. 
    - Since DDPG is off-policy and uses a deterministic target policy, this allows for the use of the Deterministic Policy Gradient theorem.
    - DDPG is an actor-critic algorithm as well; it primarily uses two neural networks, one for the actor and one for the critic. 
        - These networks compute action predictions for the current state and generate a temporal difference (TD) error signal each time step. 
    - ACTOR INPUT:
        - Current State
    - ACTOR OUTPUT:
        - Single real value representing an action chosen from a continous action space. 
    - CRITICS OUTPUT: 
        - the estimated Q-value of the current state and of the action given by the actor.     
    - The deterministic policy gradient theorem provides the update rule for the weights of the actor network. 
    - The critic network is updated from the gradients obtained from the TD error signal. 
      
+ *Possible issues in DDPG*
    - In general, training and evaluating your policy and/or value function with thousands of temporally-correlated simulated trajectories leads the the introduction of enormous amounts of variance in your approximation of the true Q-function (the critic). 
        - The TD error signal is excellent at compounding the variance introduced by your bad preditions over time. 
        - It is highly suggested to use a replay buffer to store the _experiences_ of the agent during the training, and then randomly sample experiences to use for learning in order to break up the temporal correlations within different training episodes. This technique is known as *experience replay*. 
    - Directly updating your actor and critic neural network weights with the gradients obtained from the TD error signal that was computed from both your replay buffer and the output of the actor and critic networks cause your learning algorithm to diverge(or to not learn at all).
        - It was recently discovered that using a set of *target networks* to generate the *targets* for your TD error computation regularizes your learning algorithm and increases stability. 
        - Accordingly, here are the equations for the TD target:
    - *![TD target and loss function](TD_target_and_loss_function.jpg)*
         - Here a minibatch of size $N$ has been sampled from the replay buffer, with the $i$ index referring to the i'th sample. 
         - The target for the TD error computation, $y_i$, is computed from the sum of the immediate reward and the outputs of the target actor and critic networks, having weights $\theta^{\mu'}$ and $\theta^{\Q'}$ respectively. 
        - Then the critic loss can be computed w.r.t the output $Q(s_i, a_i| \theta^{Q})$ of the critic network for the i'th sample. 
        - The weights of the critic network can be updated with the gradients obtained from the loss function in the second equation in the picture. 
        - The actor network is updated with the Deterministic Policy Gradient.
    - The stocastic _policy gradient_  $\nabla_{\theta}\mu(a|s,\theta)$ which is the gradient of the policy's performance, is equivalent to the deterministic policy gradient, which is given by:
        - 
        
