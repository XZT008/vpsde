

# VPSDE
Implementation(mostly from original implementation) and explanation of VPSDE
# TODOS
- [x] Foward Process
- [x] Predictor-Corrector sampling (pc sampling) 
- [x] Train and sample script with MNIST
- [x] ODE sampling
- [x] Likelihood estimation
- [x] BPD evaluation
- [x] Parameter tuning with Ray


# Features
**To tune hyperparameters:**
```
python tune_script.py
```
You can modify tune_config in tune_script.py if you want to use different parameter range, or add/remove hyperparameter search space. You need to set self.sampler=None during tuning if you are only interested in BPD.

For this script, I tuned over learning rate, batch size, number of resblock in Unet, channel multiplier of Unet, and sampling eps which is used to estimate bits per dim. 

**To train and sample:**
```
python run_script.py
```
You can set hyperparameters in config.py. If not, it is set to be the value used in original paper. Note that you should set self.sampler in config.py to 'ode', 'pc', or 'both', if you want to generate sample every epoch during training and testing.

# Reference:
- **Original paper:**[Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/pdf/2011.13456.pdf)
- [Official implementation](https://github.com/yang-song/score_sde_pytorch)
- **Unet:**  [Score network](https://github.com/CW-Huang/sdeflow-light/blob/main/lib/models/unet.py)

# Explanation:
## VP Forward process (Adding noise 0->T):
The two base equations are: 
$$x_{i}=\sqrt{1-\beta_{i}} x_{i-1} + \sqrt{\beta_{i}} z_{i-1}, \text{where } 0 < \beta_{1}, ..., \beta_{N} < 1$$
$$d_{x} = -\frac{1}{2}\beta(t)xd_{t}+\sqrt{\beta(t)}d_{w}, \text{where } \beta(t)=\beta_{min}+\frac{\beta_{max}-\beta_{min}}{T}t$$

The first equation is from DDPM, where it assumes noise levels are discrete. The second equation is more general and noise level is assumed to be continuous. In [Official implementation](https://github.com/yang-song/score_sde_pytorch), it chooses $T=1$ and $t \in [0+\epsilon, T]$ . The reason for adding $\epsilon$ is to avoid $t=0$, since $P_{(0)}(x) = P_{data}(x)$, which is our objective.

With vpsde, the author formulated transition probability $P_{0t}(x(t)|x(0)) = \mathcal{N}(x(t);x(0)e^{-\frac{1}{2}\int_{0}^{t} \beta(s)\,ds}, I - Ie^{-\int_{0}^{t} \beta(s)\,ds})$ (Appendix B, eqs 29). Since $\beta(s)$ is given, we can solve the integral

$$
\begin{align*}
\int_{0}^{t} \beta(s)\,ds &= \int_{0}^{t} \beta_{min}+\frac{\beta_{max}-\beta_{min}}{T}s\,ds \\
&= \frac{1}{2}s^{2}(\beta_{max}-\beta_{min})+\beta_{min}s \Big|_{0}^{t} \\
&= \frac{1}{2}t^{2}(\beta_{max}-\beta_{min})+t*\beta_{min}
\end{align*}
$$

Hence,  $$P_{0t}(x(t)|x(0)) = \mathcal{N}(x(t);e^{-\frac{1}{4}t^2(\beta_{max}-\beta_{min})-\frac{1}{2}t\beta_{min}}\,x(0), I - Ie^{-\frac{1}{2}t^2(\beta_{max}-\beta_{min})-t\beta_{min}})$$
With transition probability, we can sample perturbed data, given $x\sim x(0)$ and $t$. I implement perturbation function as follow (marginal_prob_mean_std() is a function to calculate transition prob):
```
def perturb(self, x):  
    batch_size = x.shape[0]  
    t = torch.rand(batch_size).cuda() * (self.T - self.eps) + self.eps  
    z = torch.randn_like(x).cuda()  
    mean, std = self.marginal_prob_mean_std(x, t)  
    x_tilda = mean + std.view(-1, 1, 1, 1) * z  
    return x_tilda, t, z, mean, std
```
## Training (DSM)
DSM objective is pretty clear as wrote in paper:

$$J(\theta)=\mathbb{E}_{t\sim \mathcal{U}(0, T)} [\lambda(t) \mathbb{E}_{\mathbf{x}(0) \sim p_0(\mathbf{x})}\mathbf{E}_{\mathbf{x}(t) \sim p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))}[ \|s_\theta(\mathbf{x}(t), t) - \nabla_{\mathbf{x}(t)}\log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))\|_2^2]]$$

So we sample $t$ uniformly from $[0+\epsilon, 1]$, $x$ from dataset ($P_{(0)}(x)=P_{data}$), then sample perturbed data from transition probability. Then calculate squared L2 loss with score and score estimation (Unet). $\lambda(t)$ is the weighting function and it was explained in section 4.2 [SMLD](https://arxiv.org/pdf/1907.05600.pdf). The value of $\lambda(t)$ is set to be proportional to $\frac{1}{\mathbb{E}[\|\nabla_{\mathbf{x}}\log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0)) \|_2^2]}$.  

Now let's derive $\nabla_{\mathbf{x}(t)}\log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))$. For simplicity, we reparameterize $P_{0t}(x(t)|x(0)) = \mathcal{N}(x(t);\mu x(0),\sigma^2)$. 


$$
\begin{align*}
\nabla_{\mathbf{x}(t)}\log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))&=\nabla_{\mathbf{x}(t)}\log \mathcal{N}(x(t);\mu x(0),\sigma^2) \\
&=\nabla_{\mathbf{x}(t)}\log [\frac{1}{\sigma \sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x(t)-\mu x(0)}{\sigma})^2}] \\
&= \nabla_{\mathbf{x}(t)} -\frac{1}{2}(\frac{x(t)-\mu x(0)}{\sigma})^2 \\
& = -\frac{x(t)-\mu x(0)}{\sigma ^2}
\end{align*}  
$$

By recalling how we performed data perturbation in perturb function, 
```
x_tilda = mean + std.view(-1, 1, 1, 1) * z  
```
Hence we have derived:

$$
\begin{align*}
\nabla_{\mathbf{x}(t)}\log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0)) &= -\frac{\sigma z}{\sigma ^2} = -\frac{z}{\sigma} \, \text{,where }z \sim \mathcal{N}(0, I) \\
&\sim \mathcal{N}(0, \frac{I}{\sigma^2})
\end{align*}  
$$

With the property of $E[X^2]=V[X]+(E[X])^2$, we can derive that $\lambda(t) \propto to \frac{1}{\mathbb{E}[\|\nabla_{\mathbf{x}}\log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0)) \|_2^2]}= \sigma^2$. Now we can evaluate DSM objective with MC method.

$$
\begin{align*}
J(\theta)&=\mathbb{E}_{t\sim \mathcal{U}(0, T)} [\lambda(t) \mathbb{E}_{\mathbf{x}(0) \sim p_0(\mathbf{x})}\mathbf{E}_{\mathbf{x}(t) \sim p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))}[ \|s_\theta(\mathbf{x}(t), t) - \nabla_{\mathbf{x}(t)}\log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))\|_2^2]] \\
&=\mathbb{E}_{t\sim \mathcal{U}(0, T)} [ \mathbb{E}_{\mathbf{x}(0) \sim p_0(\mathbf{x})}\mathbf{E}_{\mathbf{x}(t) \sim p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))}[ \lambda(t)\|s_\theta(\mathbf{x}(t), t) + \frac{z}{\sigma}\|_2^2]] \\
&= \frac{1}{N} \sum_{i=0}^{N}[ \mathbb{E}_{\mathbf{x}(0) \sim p_0(\mathbf{x})}\mathbf{E}_{\mathbf{x}(t_i) \sim p_{0t_{i}}(\mathbf{x}(t_{i}) \mid \mathbf{x}(0))}[(s_\theta(\mathbf{x}(t_i), t_i)\, \sigma_i + z)^2]]
\end{align*}  
$$

Here is my implementation for DSM loss:
```
def forward(self, x):  
    x_tilda, t, z, mean, std = self.perturb(x)  
    normed_score = self.score_func(x_tilda, t) / std.view(-1, 1, 1, 1)  
    return normed_score, std, t, z

def dsm_loss(self, x):  
    normed_score, std, t, z = self(x)  
    dsm_loss = torch.mean(torch.sum((normed_score * std.view(-1, 1, 1, 1) + z)**2, dim=(1, 2, 3)))  
    return dsm_loss
```

## Sampling with Reverse SDE (Predictor-Corrector Methods)
The equation for reverse SDE is: $dx=[f(x,t)-g(t)^{2}\nabla_{\mathbf{x}}\log P_{t}(x)]dt+g(t)d\bar{w}$. Note $dt$ here is negative. Now let's replace $dt$ with $-\Delta t$, where $\Delta t$ is positive. And $dw$ with $z \sim \mathcal{N}(0, \Delta tI)$. The resulting reverse SDE is: $$x_{t-\Delta t}=x_t - \Delta t[f(x,t)-g(t)^{2}\nabla_{\mathbf{x}}\log P_{t}(x)]dt+g(t)\sqrt{\Delta t}z_t$$

**I'm not quite certain about why $dw$ can be replaced this way. My understanding is that, from eqs(24) and eqs(25) in Appendix B, $\sqrt {\beta(t)\Delta t}z(t)$ converges to $\sqrt{\beta (t)}dw$, where $z(t)$ is standard Gaussian. Hence $dw \sim \mathcal{N}(0, \Delta tI)$.**

With this equation(predictor), we can do sampling already. Since $f(x,t)$, $g(t)$ are given, and we can sample $x_T$ from our prior, $\Delta t$ is time step defined by us. However, if we want to improve sample quality, we can combine predictor with corrector. As far as I understand, for every $x_{t-\Delta t}$ we generate, we use annealed Langevin dynamics(corrector) to improve its quality. Detail of corrector algorithm can be found in Appendix G, algo 5. For more detail about Langevin dynamics, please read through [SMLD](https://arxiv.org/pdf/1907.05600.pdf).

## Sampling with ODE
For every SDE: $dx=f(x,t)dt+g(t)dw$, there exists an associated ODE: $dx=[f(x,t)-\frac{1}{2}g(t)^{2}\nabla_{\mathbf{x}}\log P_{t}(x)]dt$. The good thing of ODE is that it can be solved numerically, and reverse ODE is the same as forward ODE. You can just through it in scipy.integrate.solve_ivp. The ODE sample quality is not as good as PC sampling method, since solve_ivp is just an approximation of $X(0)$, and ODE is also an approximation of SDE.

**Some implementation detail of ODE sampling:**
scipy.integrate.solve_ivp need a callable function $\frac{dy}{dt}=f(t,y)$, a time span (t0, tf), where the solver start from t=t0 and integrates until it reaches to t=tf. So our reverse process time span is (T, eps=1e-3). Note that this eps should be greater than $\epsilon$ in our forward process. The final output of solver.y is values of solution at $t$, $t\in [eps, T]$. 

Also solve_ivp only takes 1D ndarry. So remember to transform and reshape torch.tensor to 1D ndarray.

## Log likelihood estimation
With the associated ODE: $dx=\bar{f_{\theta}}(x,t)dt$, we can compute log likelihood with:
$$\log p_{0}(x(0))=\log p_{T}(x(T))+\int_{0}^{T}\nabla \cdot \bar{f_{\theta}}(x,t)dt$$
And $\nabla \cdot \bar{f_{\theta}}(x,t)=E_{\epsilon \sim \mathcal{N}(0, I)}[\epsilon^{T}\nabla\bar{f_{\theta}}(x,t)\epsilon]$. With autograd, $\nabla\bar{f_{\theta}}(x,t)\epsilon$ can be evaluated and the integral can be solved by ODE solver as well.

How? Remember what we throw into the solver is $\frac{dy}{dt}=f(t,y)$, and it can be rewrite into $dy=f(t,y)dt$. If we take integral on both side, we would have $\int_{0}^{T}dy=\int_{0}^{T}\nabla \cdot \bar{f}_{\theta}(y,t)dt=y(T)-y(0)$. And if we set $y(0)=0$, solver will output $y(T)$ directly. (I change x to y in order to follow Scipy notation).
