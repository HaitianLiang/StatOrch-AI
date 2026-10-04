# StatOrch-AI：基于原始数据、学习式规划与统计认证的自适应工具编排

## 1. 问题设定

### 1.1 原始输入与目标参数

给定任务 \(u\)，原始输入为
\[
\mathcal R_u
=
\bigl(
D_{S,u},X_{T,u},f_{S,u},\mathcal M_u
\bigr),
\]
其中
\[
D_{S,u}
=
\{(x_i^S,y_i^S)\}_{i=1}^{n_S},
\qquad
X_{T,u}
=
\{x_j^T\}_{j=1}^{n_T},
\]
分别为有标签源数据与无标签目标数据，\(f_{S,u}\) 为源模型，\(\mathcal M_u\) 包含变量定义、类别编码、数据来源、模型训练记录及预处理规范。

本文以给定目标批次的类别比例估计为主要实例：
\[
\beta_u
=
\left(
\frac1{n_T}\sum_{j=1}^{n_T}\mathbf 1\{y_j^T=k\}
\right)_{k=1}^{K}
\in\Delta^{K-1},
\]
其中
\[
\Delta^{K-1}
=
\left\{
b\in\mathbb R_{\ge0}^{K}:
\sum_{k=1}^{K}b_k=1
\right\}.
\]

目标标签 \(Y_{T,u}\) 在在线执行阶段不可访问，仅用于离线训练监督与独立实验评估。任务的目标总体或目标批次在初始化时固定，后续数据获取不得隐式改变目标参数的定义。

### 1.2 策略、轨迹与输出

令 \(\mathcal A\) 为注册工具集合，\(H_t\) 为截至时刻 \(t\) 的可观测历史。控制策略满足
\[
a_t\sim\pi_\Theta(\cdot\mid H_t),
\qquad
a_t\in\mathcal A_t^{\mathrm{exec}},
\]
其中 \(\mathcal A_t^{\mathrm{exec}}\) 为当前可执行动作集合。

执行轨迹定义为
\[
\tau
=
(a_0,o_1,a_1,o_2,\ldots,a_{T-1},o_T),
\]
其中 \(o_{t+1}\) 为真实工具执行结果。轨迹由策略与实际观测共同生成，不作为任务输入。

系统输出为
\[
\mathcal O_u
=
\bigl(
s_{\mathrm{out}},
\widehat\beta_u,
\mathcal C_{\mathrm{root}},
\mathcal E_{\mathrm{sup}},
\mathcal B_T,
\tau,
\mathcal U_T,
\mathcal K_T
\bigr),
\]
分别对应输出状态、最终估计、根证书、支持证据子图、分支账本、执行轨迹、残余不确定性与资源账本。

### 1.3 执行约束

任务初始化时固定
\[
\Gamma
=
\bigl(
\varepsilon,\delta,B_0,N_{\max},H_{\mathrm{plan}},
\lambda_c,\lambda_h,\lambda_\perp,\kappa
\bigr).
\]

其中，\(\varepsilon\) 为认证容差，\(\delta\) 为总体错误概率预算，\(B_0\) 为资源预算，\(N_{\max}\) 为最大执行轮数，\(H_{\mathrm{plan}}\) 为单轮规划深度。

单轮规划深度与完整执行长度分别约束：
\[
H_{\mathrm{plan}}\ll N_{\max}.
\]
每轮仅执行规划中的第一个动作，随后基于真实观测重新规划。完整执行长度由认证条件、资源预算与技术终止条件共同决定。StatOrch_AI_Current_Conversatio…

---

## 2. 原始数据处理与初始化

### 2.1 数据登记与一致性检查

为每个样本、变量、数据分区、模型与预处理对象分配不可变标识。初始化检查包括数据维度、类别编码、缺失模式、样本重复、模型输入格式及源模型训练数据归属。

原始数据保持只读。所有变换、筛选、拟合及适配结果均以版本化派生对象保存：
\[
\operatorname{Artifact}
=
(\mathrm{id},\mathrm{parent},\mathrm{version},
\mathrm{transform},\mathrm{sample\_ids},\mathrm{seed}).
\]

不满足输入契约的任务返回技术未决状态，并记录失败字段与对应检查结果。

### 2.2 数据使用权限与审计分区

源数据按用途登记为
\[
D_S
=
D_S^{\mathrm{fit}}
\cup
D_S^{\mathrm{ref}}
\cup
D_S^{\mathrm{audit}},
\]
目标数据登记为
\[
X_T
=
X_T^{\mathrm{work}}
\cup
X_T^{\mathrm{audit}}.
\]

需要独立校准的工具仅使用未参与对应模型拟合的源样本，或使用按预先登记规则构造的折外预测。

审计数据在被释放前不得进入控制器表示、工具选择或模型更新。审计批次的选择规则、释放时刻及样本标识进入证据账本。重复访问已经释放的数据不计为新增独立观测。

### 2.3 缺失处理与数值变换

令 \(m_{ir}\) 为样本 \(i\) 的第 \(r\) 个变量的观测掩码。数值变量采用
\[
\widetilde x_{ir}
=
\frac{\operatorname{Imp}_{\eta}(x_{ir},m_{ir})-\mu_r}
{\max(s_r,\epsilon_{\mathrm{num}})},
\]
其中 \(\eta,\mu_r,s_r\) 按预先登记规则在允许的源数据上拟合。

缺失掩码与变换后数值共同输入表示网络。类别变量使用固定词表；未知类别与缺失类别使用独立编码。

源模型调用遵循其原始预处理契约：
\[
p_i=f_S(T_S(x_i)).
\]
控制器使用的数值变换不得未经登记替换源模型的输入变换 \(T_S\)。

### 2.4 初始数据访问

在预算约束下，按固定初始化规则释放样本集合
\[
I_{S,0}\subseteq D_S^{\mathrm{fit}}\cup D_S^{\mathrm{ref}},
\qquad
I_{T,0}\subseteq X_T^{\mathrm{work}}.
\]

对已释放样本计算源模型预测及允许访问的中间表示：
\[
p_i^S=f_S(T_S(x_i^S)),
\qquad
p_j^T=f_S(T_S(x_j^T)),
\]
\[
z_i^S=\operatorname{Feat}_{f_S}(x_i^S),
\qquad
z_j^T=\operatorname{Feat}_{f_S}(x_j^T).
\]

无法访问中间表示时，使用缺失标记。初始数据读取、模型推理与表示提取的实际成本计入 \(C_{\mathrm{init}}\)。

初始化不强制执行统计检验、模型适配或比例估计。

---

## 3. 原始问题表示与交互状态

### 3.1 样本级表示

源样本与目标样本分别编码为
\[
r_i^S
=
\phi_\theta
\left(
\widetilde x_i^S,
m_i^S,
e(y_i^S),
p_i^S,
z_i^S,
e_S
\right),
\]
\[
r_j^T
=
\phi_\theta
\left(
\widetilde x_j^T,
m_j^T,
e_{\mathrm{unlabeled}},
p_j^T,
z_j^T,
e_T
\right).
\]

其中 \(e_S,e_T\) 为域标识，\(e_{\mathrm{unlabeled}}\) 为标签不可用标识。集合编码器对样本排列保持不变性或等变性。[NeurIPS 会议论文集](https://proceedings.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html?utm_source=chatgpt.com)

### 3.2 源—目标联合表示

分别构造
\[
Z_S
=
E_{\theta,S}\bigl(\{r_i^S\}_{i\in I_{S,0}}\bigr),
\qquad
Z_T
=
E_{\theta,T}\bigl(\{r_j^T\}_{j\in I_{T,0}}\bigr),
\]
并通过交叉注意力形成联合表示
\[
Z_{ST}
=
\operatorname{CrossAttn}_\theta(Z_S,Z_T).
\]

初始问题表示为
\[
Z_0
=
F_\theta
\bigl(
Z_S,Z_T,Z_{ST},e(f_S),e(\mathcal M_u)
\bigr).
\]

原始分布表示与后续统计证据采用独立编码通道，并在历史编码器中融合。设计AI决策续集

### 3.3 工具执行事件

每个真实执行事件记录为
\[
\mathsf e_{t+1}
=
\left(
a_t,o_{t+1},c_t,
b_t,v_t,v_{t+1},
\mathcal I_{t+1},
r_{t+1}
\right),
\]
其中 \(b_t\) 为分支标识，\(v_t,v_{t+1}\) 为执行前后版本，\(\mathcal I_{t+1}\) 为涉及的样本与工件标识，\(r_{t+1}\) 为执行状态。

事件编码为
\[
e_{t+1}
=
E_{\mathrm{event},\theta}(\mathsf e_{t+1}).
\]

统计证据事件同时记录工具类型、数值、误差估计、有效样本量、缺失标记、数据来源及适用范围。来自同一数据的重采样结果与重复计算结果保留共同来源标识。

### 3.4 完整执行状态

系统维护
\[
\Xi_t
=
\left(
b_t,
\mathcal B_t,
\mathcal D_t,
\mathcal E_t,
\mathcal W_t,
H_t,
B_t,
n_t,
\mathcal K_t
\right),
\]
其中：

\[
\mathcal B_t:
\text{分支及模型快照集合},
\qquad
\mathcal D_t:
\text{已生成的候选结果集合},
\]
\[
\mathcal E_t:
\text{真实证据账本},
\qquad
\mathcal W_t:
\text{可信世界集合},
\]
\[
B_t:
\text{剩余资源预算},
\qquad
n_t=N_{\max}-t.
\]

控制器历史表示为
\[
h_t
=
E_{\mathrm{hist},\theta}
\left(
Z_0,
e_{1:t},
E(\mathcal B_t),
E(\mathcal D_t),
E(\mathcal W_t),
B_t,n_t
\right).
\]

历史编码器采用因果掩码。未执行分支的真实结果、尚未释放的数据及目标标签不得进入 \(h_t\)。

神经表示 \(h_t\) 用于预测与规划；统计认证直接读取具有来源记录的证据与原始工件。

---

## 4. 统一工具空间与执行接口

### 4.1 工具注册

每个工具登记为
\[
\mathsf T_a
=
\left(
\mathrm{id}_a,
\mathrm{type}_a,
\mathcal I_a,
\mathcal O_a,
\mathcal P_a,
\mathcal V_a,
\mathsf{Exec}_a
\right),
\]
其中 \(\mathcal I_a,\mathcal O_a\) 为输入与输出契约，\(\mathcal P_a\) 为前置条件，\(\mathcal V_a\) 为实现版本及参数规范。

统一工具集合为
\[
\mathcal A
=
\mathcal A_{\mathrm{observe}}
\cup
\mathcal A_{\mathrm{acquire}}
\cup
\mathcal A_{\mathrm{adapt}}
\cup
\mathcal A_{\mathrm{infer}}
\cup
\mathcal A_{\mathrm{audit}}
\cup
\mathcal A_{\mathrm{repair}}
\cup
\mathcal A_{\mathrm{control}}.
\]

诊断、干预、估计与审计属于工具类型，不构成固定执行阶段。

动作实例表示为
\[
a_t
=
(\mathrm{tool\_id},\mathrm{args},
\mathrm{branch\_id},\mathrm{input\_versions},\mathrm{seed}).
\]

### 4.2 当前合法动作

定义
\[
\mathcal A_t^{\mathrm{legal}}
=
\left\{
a\in\mathcal A:
\mathcal P_a(\Xi_t)=1,\;
\overline c_t(a)\le B_t-B_{\mathrm{reserve}}
\right\},
\]
其中 \(B_{\mathrm{reserve}}\) 为最终审计与输出所保留的预算。

合法性检查覆盖数据权限、模型兼容性、工件版本、参数范围、数值条件、资源限制与回退能力。

重复工具调用仅在输入版本、参数、样本集合或随机试验发生实质变化时产生新的执行节点。缓存命中不增加独立证据计数。

### 4.3 统一执行结果

工具执行接口为
\[
(o_{t+1},\mathcal A\!r_{t+1},c_t,r_{t+1})
=
\mathsf{Exec}_{a_t}(\Xi_t),
\]
其中 \(\mathcal A\!r_{t+1}\) 为新增或修改的工件。

对于估计工具 \(q\)，
\[
\widehat\beta_{t+1}^{(q)}
=
\mathsf Q_q
\left(
D_S^{\mathrm{allowed}},
X_T^{\mathrm{allowed}},
f^{(b_t)},
\mathcal A\!r_t;
\mathrm{args}_q
\right).
\]

有效比例估计满足
\[
\widehat\beta_{t+1}^{(q)}\in\Delta^{K-1}.
\]
违反输出契约、数值发散或前置条件失效的执行返回失败状态，不生成有效候选结果。

### 4.4 工具表示

动作编码为
\[
e_a
=
E_{\mathrm{tool},\theta}
\left(
\mathrm{id}_a,
\mathrm{type}_a,
\mathcal P_a,
\mathcal I_a,
\mathcal O_a,
\mathrm{args}_a,
\widehat c_a
\right).
\]

对最终提交动作，编码中额外包含被提交候选结果及其来源分支。工具表示与历史表示共同用于动作条件预测。设计AI决策续集

---

## 5. 任务目标与方法选择准则

### 5.1 终端任务损失

对数值输出 \(d=\widehat\beta\)，定义
\[
\ell_u(d)
=
\|\widehat\beta-\beta_u\|_1.
\]

对无数值结论的终止输出 \(d=\perp\)，定义
\[
\ell_u(\perp)=\lambda_\perp.
\]

令 \(d_{0,u}\) 为预先登记的非适配参考程序输出。有害干预惩罚定义为
\[
H_u(\tau)
=
\mathbf 1\{\tau\text{包含干预且输出数值结果}\}
\left[
\ell_u(d_\tau)-\ell_u(d_{0,u})-\eta_h
\right]_+,
\]
其中 \(\eta_h\ge0\) 为预先固定的容忍量。

### 5.2 总成本

完整成本为
\[
C_u(\tau)
=
C_{\mathrm{init}}
+
\sum_{t=0}^{T-1}c_t,
\]
其中 \(c_t\) 包含本轮真实发生的数据访问、工具执行、模型规划、审计认证与状态维护成本。

失败调用、放弃分支与回退之前已经消耗的成本均保留。

### 5.3 路径目标

定义
\[
\boxed{
J_u(\tau)
=
\ell_u(d_\tau)
+
\lambda_cC_u(\tau)
+
\lambda_hH_u(\tau).
}
\]

该目标以最终任务误差为主要质量指标，并包含工具成本与有害干预惩罚。设计AI决策续集

令 \(\Pi_{\Gamma}\) 为满足数据权限、预算与执行约束的历史依赖策略类。策略目标为
\[
\min_{\pi\in\Pi_\Gamma}
\mathbb E_{u,\pi}[J_u(\tau_\pi)].
\]

### 5.4 剩余代价

令 \(G_u(H,d)\) 为在历史 \(H\) 下提交 \(d\) 时的终端损失及有害干预惩罚：
\[
G_u(H,d)
=
\ell_u(d)+\lambda_hH_u(H,d).
\]

对非终止动作，
\[
Q_n^\star(H,a)
=
\mathbb E
\left[
\lambda_cc(H,a,O')
+
V_{n-1}^\star(H')
\mid H,a
\right].
\]

对提交动作，
\[
Q_n^\star(H,\operatorname{FINALIZE}(d))
=
\mathbb E[G_u(H,d)\mid H].
\]

相应地，
\[
V_n^\star(H)
=
\min_{a\in\mathcal A^{\mathrm{exec}}(H)}
Q_n^\star(H,a).
\]

已发生的历史成本不重复计入剩余代价。

### 5.5 在线选择指标

模型对候选动作输出
\[
\bigl(
\widehat\mu_t(a),
\widehat\sigma_t(a)
\bigr),
\]
并构造风险调整评分
\[
\boxed{
R_t(a)
=
\widehat\mu_t(a)
+
\kappa\widehat\sigma_t(a).
}
\]

在线动作选择为
\[
\boxed{
a_t
\in
\arg\min_{a\in\mathcal A_t^{\mathrm{exec}}}
R_t(a).
}
\]

该评分为学习式规划准则。统计认证不以
\(\widehat\mu_t(a)\pm\kappa\widehat\sigma_t(a)\)
自动构造有效置信区间。

---

## 6. 世界模型、价值模型与不确定性

### 6.1 动作条件世界模型

世界模型预测
\[
p_\phi
\left(
o_{t+1},
\Delta\mathcal A\!r_{t+1},
c_t,
r_{t+1}
\mid h_t,e_{a_t}
\right).
\]

观测工具预测返回证据，干预工具预测工件变化，估计工具预测候选结果，审计工具预测审计结果，控制工具预测分支与预算状态变化。

模拟后继历史表示为
\[
\widehat h_{t+1}
=
\operatorname{Update}_{\theta}
\left(
h_t,a_t,\widehat o_{t+1},
\widehat{\Delta\mathcal A\!r}_{t+1}
\right).
\]

世界模型的预测结果仅用于规划，不进入真实证据账本。原始设计中的动作条件动态预测与显式前瞻保留为模型核心。设计AI决策续集

### 6.2 价值模型

价值模型为
\[
Q_\psi(h_t,e_a,n_t,B_t)
=
\left(
\mu_\psi(h_t,a),
s_\psi^2(h_t,a)
\right).
\]

其中 \(\mu_\psi\) 预测剩余总代价，\(s_\psi^2\) 预测训练目标或后续回报的条件变异。

候选结果、执行版本、剩余预算与剩余轮数均作为价值模型输入。

### 6.3 模型集成

采用 \(M\) 个模型组成集成：
\[
\{(p_{\phi_m},Q_{\psi_m})\}_{m=1}^{M}.
\]

集成均值与方差定义为
\[
\widehat\mu(h,a)
=
\frac1M\sum_{m=1}^{M}\mu_m(h,a),
\]
\[
\widehat\sigma_{\mathrm{ep}}^2(h,a)
=
\frac1M\sum_{m=1}^{M}
\left(
\mu_m(h,a)-\widehat\mu(h,a)
\right)^2,
\]
\[
\widehat\sigma_{\mathrm{al}}^2(h,a)
=
\frac1M\sum_{m=1}^{M}s_m^2(h,a),
\]
\[
\widehat\sigma^2(h,a)
=
\widehat\sigma_{\mathrm{ep}}^2(h,a)
+
\widehat\sigma_{\mathrm{al}}^2(h,a).
\]

模型集成与随机轨迹传播共同用于规划中的预测不确定性评估。[NeurIPS 会议论文集](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html?utm_source=chatgpt.com)

---

## 7. Paper A 认证接口与证据状态

### 7.1 可信世界

令
\[
\omega
=
(M,\beta,\gamma,\nu,\ldots)\in\Omega
\]
表示包含模型结构、目标参数、分布差异及其他未知量的候选世界。

认证模块读取真实证据账本：
\[
\mathsf A
\left(
\mathcal E_t,\Omega,\varepsilon,\delta
\right)
=
\left(
\mathcal W_t,
\mathcal S_t,
\overline{\mathcal B}_t,
\mathcal Q_t,
\mathcal P_t
\right),
\]
分别返回可信世界集合、认证状态、经过验证的风险上界、待解决证据需求及证明记录。

拟合、预测、结构、决策与可辨识证据共同约束 \(\mathcal W_t\)。神经预测结果与未经有效性验证的诊断数值不直接排除候选世界。StatOrch_AI_Current_Conversatio…

### 7.2 时间一致覆盖条件

统计保证以认证模块满足
\[
\boxed{
\mathbb P_{\omega^\star}
\left(
\forall t\ge0:
\omega^\star\in\mathcal W_t
\right)
\ge1-\delta
}
\]
为前提，其中 \(\omega^\star\in\Omega\)。

该条件须覆盖自适应工具选择、重复检查、数据依赖分支选择与随机停止。时间一致置信集合用于支持任意停止时刻下的有效推断。[arXiv](https://arxiv.org/abs/1810.08240?utm_source=chatgpt.com)

对第 \(j\) 个新登记的统计声明分配
\[
\delta_j
=
\frac{6\delta}{\pi^2j^2},
\qquad
\sum_{j=1}^{\infty}\delta_j\le\delta.
\]

每个声明须满足相应的条件有效性要求。分支切换、模型修复与重新进入不得重置总体错误概率预算。

### 7.3 最终估计认证

对于候选估计 \(d=\widehat\beta\)，定义
\[
E_t(d)
=
\sup_{\omega\in\mathcal W_t}
\|d-\beta(\omega)\|_1.
\]

认证模块计算经过验证的上界
\[
\overline E_t(d)\ge E_t(d).
\]

最终估计认证条件为
\[
\boxed{
\mathcal W_t\ne\varnothing,
\qquad
\overline E_t(d)\le\varepsilon.
}
\]

近似优化器仅返回候选值、下界或未验证解时，不构成上述认证。

### 7.4 相对决策认证

对预先登记的比较类 \(\mathcal D_{\mathrm{ref}}\)，定义
\[
B_t(d)
=
\sup_{\omega\in\mathcal W_t}
\left[
L(d,\omega)
-
\inf_{d'\in\mathcal D_{\mathrm{ref}}}L(d',\omega)
\right].
\]

若
\[
\overline B_t(d)\ge B_t(d),
\qquad
\overline B_t(d)\le\varepsilon,
\]
则输出相对于 \(\mathcal D_{\mathrm{ref}}\) 的近优决策证书。该认证采用可信世界上的最坏情形决策遗憾。StatOrch_AI_Current_Conversatio…

### 7.5 认证范围

每个证书显式登记
\[
\mathrm{scope}(\mathcal C)
\in
\{
\text{工具适用性},
\text{局部决策},
\text{最终估计},
\text{完整策略}
\}.
\]

完整策略近优性须另外验证
\[
\sup_{\omega\in\mathcal W_t}
\left[
J_\omega(\pi)
-
\inf_{\pi'\in\Pi_{\mathrm{ref}}}J_\omega(\pi')
\right]
\le\varepsilon_{\mathrm{prog}}.
\]

局部动作认证与最终估计认证均不自动推出完整策略近优性。

---

## 8. 离线交互图与模型训练

### 8.1 任务划分

离线任务划分为
\[
\mathcal U
=
\mathcal U_{\mathrm{train}}
\cup
\mathcal U_{\mathrm{val}}
\cup
\mathcal U_{\mathrm{test}}.
\]

同一原始任务产生的全部分支、重复执行、重采样结果与共享源数据的派生任务按照预先登记的分组规则划分，禁止跨集合泄漏。

模型参数、归一化参数、评分系数及停止超参数仅使用训练集与验证集确定。

### 8.2 真实交互图

对每个训练任务构造
\[
\mathcal G_u=(\mathcal V_u,\mathcal E_u).
\]

节点保存真实执行状态，边保存
\[
\left(
H_t,a_t,o_{t+1},
\Delta\mathcal A\!r_{t+1},
c_t,r_{t+1},H_{t+1}
\right).
\]

所有监督用工具后果均来自真实执行。反事实分支由同一父快照独立恢复后运行，模型参数、优化器状态、数据版本与随机数状态一并恢复。

节点包含累计执行轮数与剩余预算；回退后的节点与原父节点保留不同的历史与资源状态。

交互图包含合法动作覆盖、当前策略访问状态及恢复分支。未执行的工具后果不得登记为已观测结果。离线交互图保留原设计中用于动态预测、价值学习与排序监督的功能。设计AI决策续集

### 8.3 终端监督

提交候选结果 \(d_i\) 时，监督目标为
\[
y_i^{\mathrm{term}}
=
\ell_{u_i}(d_i)
+
\lambda_hH_{u_i}(H_i,d_i).
\]

目标标签仅用于计算该监督值。标签及其派生量不得写入控制器历史。

逐任务事后最优路径
\[
\tau_u^{\mathrm{hind}}
\in
\arg\min_{\tau\in\mathcal T(\mathcal G_u)}J_u(\tau)
\]
仅定义为已执行交互图中的评估基准。

### 8.4 非终端价值目标

令 \(\bar\psi\) 为冻结目标网络。非终端转移采用
\[
y_i^{(k)}
=
\lambda_cc_i
+
\min_{a'\in\mathcal A^{\mathrm{exec}}(H_i')}
\mu_{\bar\psi}^{(k-1)}(h_i',a').
\]

模型通过条件回归逼近
\[
Q_n^\star(H,a)
=
\mathbb E
\left[
\lambda_cc+
\min_{a'}Q_{n-1}^\star(H',a')
\mid H,a
\right].
\]

后续动作选择只能依赖后继可观测历史。不得使用当前任务隐藏标签或未访问分支的真实损失决定 Bellman 续接动作。

### 8.5 动态损失

对工具返回结果采用类型对应的似然模型：
\[
\mathcal L_{\mathrm{dyn}}
=
-\mathbb E_{\mathcal D}
\log
p_\phi
\left(
o',\Delta\mathcal A\!r,c,r
\mid h,a
\right).
\]

对潜在后继状态加入
\[
\mathcal L_{\mathrm{latent}}
=
\mathbb E_{\mathcal D}
\left\|
\widehat h'
-
\operatorname{sg}
\left(
E_{\bar\theta}(H')
\right)
\right\|_2^2.
\]

不同类型的连续输出按训练集尺度归一化；缺失或无效字段不参与对应损失项。

### 8.6 价值损失

价值均值采用
\[
\mathcal L_Q
=
\mathbb E_{\mathcal D}
\left[
\bigl(\mu_\psi(h,a)-y\bigr)^2
\right].
\]

### 8.7 排序损失

对同一可观测状态下的候选动作 \(a,b\)，令
\[
s_{ab}
=
\operatorname{sgn}(\overline y_b-\overline y_a),
\]
其中 \(\overline y_a,\overline y_b\) 为相应价值目标的估计均值。

仅对
\[
|\overline y_b-\overline y_a|>\eta_{\mathrm{rank}}
\]
的动作对构造
\[
\mathcal L_{\mathrm{rank}}
=
\mathbb E
\log
\left[
1+
\exp
\left(
-\frac{
s_{ab}\bigl(\mu_\psi(h,b)-\mu_\psi(h,a)\bigr)
}{T_{\mathrm{rank}}}
\right)
\right].
\]

### 8.8 不确定性损失

对预测方差 \(s_\psi^2>0\)，采用
\[
\mathcal L_{\mathrm{unc}}
=
\mathbb E
\left[
\frac{(y-\mu_\psi)^2}
{2(s_\psi^2+\epsilon_{\mathrm{var}})}
+
\frac12\log(s_\psi^2+\epsilon_{\mathrm{var}})
\right].
\]

预测区间的经验覆盖与代价排序在独立验证任务上评估，不替代第 7 节的统计有效性条件。

### 8.9 掩码证据损失

在当前历史内随机遮蔽已经可观测的证据字段，定义
\[
\mathcal L_{\mathrm{mask}}
=
-\mathbb E
\log p_\theta
\left(
o_{\mathrm{masked}}
\mid
H_{\mathrm{visible}}
\right).
\]

未来观测、隐藏标签及其他分支未访问结果不作为可见上下文。

### 8.10 联合训练

总体训练目标为
\[
\boxed{
\begin{aligned}
\mathcal L_{\mathrm{train}}
={}&
\alpha_Q\mathcal L_Q
+\alpha_{\mathrm{rank}}\mathcal L_{\mathrm{rank}}
+\alpha_{\mathrm{dyn}}\mathcal L_{\mathrm{dyn}}\\
&+\alpha_{\mathrm{latent}}\mathcal L_{\mathrm{latent}}
+\alpha_{\mathrm{unc}}\mathcal L_{\mathrm{unc}}
+\alpha_{\mathrm{mask}}\mathcal L_{\mathrm{mask}}.
\end{aligned}
}
\]

训练依次执行表示与动态预训练、价值拟合、联合更新及当前策略的数据聚合。目标网络按固定规则更新。最终参数由独立验证任务上的任务损失、执行成本与输出覆盖共同选定。

在线执行期间冻结训练参数；单任务中的历史更新与工件适配不等同于重新训练控制器。

---

## 9. 有限前瞻与持续重规划

### 9.1 条件策略树

在时刻 \(t\)，构造深度不超过 \(H_{\mathrm{plan}}\) 的候选条件策略集合
\[
\Pi_{t,H}^{\mathrm{search}}.
\]

后续动作允许依赖模拟观测：
\[
a_{t+j}
=
\pi_j(\widehat H_{t+j}),
\qquad
j=0,\ldots,H_{\mathrm{plan}}-1.
\]

搜索对象为条件策略树，而非不依赖后续观测的固定动作序列。

### 9.2 模拟回报

对候选策略 \(\pi\)、模型成员 \(m\) 与随机传播样本 \(k\)，生成
\[
Z_{\pi}^{m,k}
=
\sum_{j=0}^{L-1}
\lambda_c\widehat c_{t+j}^{m,k}
+
\begin{cases}
\widehat G^{m,k}, & \text{模拟在深度 }L\text{ 终止},\\[1mm]
\widehat V_\psi(\widehat h_{t+L}^{m,k}),&
L=H_{\mathrm{plan}}\text{ 且未终止}.
\end{cases}
\]

模拟中的工具失败、分支回退与认证未决均进入后继状态和回报。

计算
\[
\widehat\mu_t(\pi)
=
\frac1{MK}\sum_{m=1}^{M}\sum_{k=1}^{K}Z_\pi^{m,k},
\]
\[
\widehat\sigma_t^2(\pi)
=
\frac1{MK}
\sum_{m=1}^{M}\sum_{k=1}^{K}
\left(
Z_\pi^{m,k}-\widehat\mu_t(\pi)
\right)^2.
\]

### 9.3 根动作评分

对当前候选动作 \(a\)，定义
\[
\boxed{
R_t^{(H)}(a)
=
\min_{\substack{
\pi\in\Pi_{t,H}^{\mathrm{search}}\\
\pi_0=a
}}
\left[
\widehat\mu_t(\pi)
+
\kappa\widehat\sigma_t(\pi)
\right].
}
\]

选择
\[
a_t^{\mathrm{prop}}
\in
\arg\min_{a\in\mathcal A_t^{\mathrm{legal}}}
R_t^{(H)}(a).
\]

有限搜索得到的最优性范围限定于
\(\Pi_{t,H}^{\mathrm{search}}\)；未搜索策略不被登记为已排除策略。

### 9.4 证据获取价值

对仅产生观测的动作 \(a\)，在期望代价准则下，
\[
Q_t^{\mathrm{obs}}(a)
=
\lambda_c\widehat c_t(a)
+
\mathbb E_{\widehat o\sim p_\phi(\cdot\mid h_t,a)}
\left[
\min_b
\widehat Q_{t+1}(\widehat H_{t+1},b)
\right].
\]

对应的信息价值为
\[
\operatorname{EVI}_t(a)
=
\widehat V_t^{\mathrm{current}}
-
Q_t^{\mathrm{obs}}(a).
\]

同一固定数据上的诊断与重采样仅改变已计算信息和计算成本，不增加独立样本数；新增测量与新审计批次按其实际采样机制更新统计证据。

### 9.5 执行门控

动作提交后进行
\[
g_t
=
\operatorname{Gate}
\left(
a_t^{\mathrm{prop}},
\Xi_t,
\mathcal P_t
\right).
\]

门控结果包含可执行、需要补充证据、前置条件失败与技术不可执行。

可回退的内部计算在满足数据权限与执行安全条件时进入探索分支。最终提交或不可逆操作须满足相应范围的统计证书。

未通过门控的动作不执行，并登记失败原因及解除条件。规划器随后选择证据获取、修复、其他工具或其他分支。相同状态下重复遭拒且解除条件未变化的动作不重复提交。

---

## 10. 真实执行、状态更新与分支恢复

### 10.1 执行前快照

对修改模型、表示或预处理对象的动作，保存
\[
\mathsf{Snapshot}_t
=
\left(
f^{(b_t)},
T^{(b_t)},
\mathrm{optimizer},
\mathrm{rng},
\mathrm{artifact\_versions}
\right).
\]

非终止动作仅执行当前规划的第一步：
\[
(o_{t+1},\mathcal A\!r_{t+1},c_t,r_{t+1})
=
\mathsf{Exec}_{a_t}(\Xi_t).
\]

### 10.2 预算与历史更新

更新
\[
B_{t+1}=B_t-c_t,
\qquad
n_{t+1}=n_t-1,
\]
\[
H_{t+1}
=
H_t\oplus
(a_t,o_{t+1},c_t,r_{t+1}).
\]

失败调用同样更新资源账本与历史。回退不恢复已经消耗的预算。

### 10.3 工件与候选结果更新

若执行生成新工件，则登记父子关系与版本依赖。若执行生成有效候选估计，则更新
\[
\mathcal D_{t+1}
=
\mathcal D_t
\cup
\left\{
(d_{t+1},\mathrm{branch},\mathrm{version},\mathrm{provenance})
\right\}.
\]

模型或表示发生变化后，依赖旧版本的当前诊断与证书失去对新版本的直接适用性。历史记录保持不变，旧快照对应的候选结果与证书保留原适用范围。

### 10.4 证据更新与重检

真实执行结果经来源与有效性检查后进入
\[
\mathcal E_{t+1}
=
\operatorname{AppendValid}
(\mathcal E_t,o_{t+1}).
\]

认证模块重新计算
\[
(\mathcal W_{t+1},\mathcal S_{t+1},
\overline{\mathcal B}_{t+1},
\mathcal Q_{t+1},\mathcal P_{t+1})
=
\mathsf A(\mathcal E_{t+1}).
\]

模型版本变化后，仅更新已经失效且当前需要的表示与证据。额外检验由后续动作选择产生，不强制重新运行全部诊断。

### 10.5 分支状态

每个分支 \(b\) 维护状态
\[
s_t^{(b)}
\in
\left\{
\begin{aligned}
&\texttt{ACTIVE},
\texttt{CERTIFIED},
\texttt{REJECTED\_MODEL},\\
&\texttt{ABSTAIN\_STRUCTURAL},
\texttt{UNRESOLVED\_BUDGET},
\texttt{UNRESOLVED\_TECHNICAL}
\end{aligned}
\right\}.
\]

模型拒绝或结构性弃权关闭当前分支，不自动终止整个任务。系统恢复可用快照，并在其余可执行分支上重新规划。

模型修复生成具有新假设登记与新版本的新分支。原拒绝证据、已消耗成本与统计错误概率支出继续保留。

### 10.6 历史表示更新

依据真实执行与认证结果计算
\[
h_{t+1}
=
E_{\mathrm{hist},\theta}(\Xi_{t+1}),
\]
随后重新执行有限前瞻规划。上一轮未执行的模拟后续路径全部作废。

---

## 11. 停止规则与完整在线算法

### 11.1 认证提交

定义可认证候选集合
\[
\mathcal D_t^{\mathrm{cert}}
=
\left\{
d\in\mathcal D_t:
\operatorname{ValidVersion}(d)=1,\;
\mathcal W_t\ne\varnothing,\;
\overline E_t(d)\le\varepsilon
\right\}.
\]

对
\(d\in\mathcal D_t^{\mathrm{cert}}\)，允许提交动作
\(\operatorname{FINALIZE}(d)\)。

当
\[
R_t^{(H)}(\operatorname{FINALIZE}(d_t))
\le
\min_{a\in\mathcal A_t^{\mathrm{exec}}
\setminus\mathcal A_{\mathrm{final}}}
R_t^{(H)}(a)
+
\varepsilon_{\mathrm{plan}},
\]
执行认证提交。

### 11.2 预算终止

满足以下任一条件时触发预算终止：
\[
n_t=0,
\]
或
\[
B_t
<
B_{\mathrm{reserve}}
+
\min_{a\in\mathcal A_t^{\mathrm{legal}}}
\overline c_t(a).
\]

若已有有效根证书，则提交对应认证结果；否则返回未决状态，或按照预先登记的运行模式返回未认证数值结果。

### 11.3 无可认证路线终止

仅当根证明覆盖全部预先登记的可容许分支，并证明其均不满足任务规定的认证目标时，返回
\[
\texttt{CERTIFIED\mbox{-}NO\mbox{-}ROUTE}.
\]

启发式剪枝、搜索未找到候选、模型低置信度或已访问分支全部失败，不构成无路线证明。

### 11.4 技术终止

工具持续失败、认证求解超时、必要输入不可获取或全部剩余动作不可执行时，返回技术未决状态。

可信世界集合为空时禁止输出正向认证：
\[
\mathcal W_t=\varnothing
\Longrightarrow
\operatorname{CanCertify}_t=0.
\]

### 11.5 完整在线算法

**算法 1：StatOrch-AI 在线执行**

**输入：** 原始任务 \(\mathcal R_u\)，冻结模型参数 \(\Theta\)，工具注册表 \(\mathcal A\)，任务配置 \(\Gamma\)，Paper A 认证接口。

**输出：** 最终结果档案 \(\mathcal O_u\)。

1. 检查输入契约，登记数据与模型来源，建立只读原始数据及工作、参考、审计分区。

2. 拟合允许的预处理对象，释放初始化样本，计算初始表示 \(Z_0\)，登记初始化成本。

3. 初始化源模型分支、候选结果集合、证据账本、可信世界、资源账本及错误概率账本。

4. 计算历史表示 \(h_t\)，更新所有有效候选结果的根认证状态。

5. 检查认证提交、预算终止、无路线证明与技术终止条件；满足终止条件时生成结果档案并结束。

6. 根据当前数据权限、工件版本、工具前置条件与剩余预算生成 \(\mathcal A_t^{\mathrm{legal}}\)。

7. 构造有限深度条件策略树，使用世界模型传播候选工具后果，并计算 \(R_t^{(H)}(a)\)。

8. 按评分顺序提交候选动作。未通过门控时登记解除条件，并改选证据获取、修复或其他分支动作。

9. 对获准的修改动作建立快照，真实执行当前动作，记录工具返回、工件变化、实际成本与执行状态。

10. 更新资源账本、候选结果、版本依赖、真实证据账本与 Paper A 状态；必要时关闭当前分支并恢复其他快照。

11. 使用真实返回更新 \(h_{t+1}\)，令 \(t\leftarrow t+1\)，返回步骤 4。

---

## 12. 终止性、认证保证与收敛条件

### 12.1 有限终止

**命题 1（有限执行终止）。**  
若每轮执行、规划与认证调用均具有有限超时，且 \(N_{\max}<\infty\)，则算法在至多 \(N_{\max}\) 个非终止轮次后返回结果档案。

若进一步存在 \(c_{\min}>0\)，使每个非终止轮次实际成本满足
\[
c_t\ge c_{\min},
\]
则
\[
T
\le
\min
\left\{
N_{\max},
\left\lfloor
\frac{B_0-C_{\mathrm{init}}-B_{\mathrm{reserve}}}
{c_{\min}}
\right\rfloor
\right\}.
\]

回退、重新进入与分支切换不改变该界。

### 12.2 自适应停止下的输出保证

**命题 2（最终估计认证）。**  
设第 7.2 节的时间一致覆盖条件成立，且认证上界满足
\[
\overline E_t(d)
\ge
\sup_{\omega\in\mathcal W_t}
\|d-\beta(\omega)\|_1.
\]
则对算法产生的任意随机停止时刻 \(T\)，
\[
\boxed{
\mathbb P
\left(
s_{\mathrm{out}}=\texttt{CERTIFIED\mbox{-}ACT},
\;
\|\widehat\beta_T-\beta_u\|_1>\varepsilon
\right)
\le\delta.
}
\]

**证明。** 在事件
\[
\{\forall t:\omega^\star\in\mathcal W_t\}
\]
上，认证提交满足
\[
\|\widehat\beta_T-\beta_u\|_1
\le
\sup_{\omega\in\mathcal W_T}
\|\widehat\beta_T-\beta(\omega)\|_1
\le
\overline E_T(\widehat\beta_T)
\le\varepsilon.
\]
结论由覆盖事件的概率界得到。 \(\square\)

### 12.3 认证收敛

**命题 3（条件性证据收敛）。**  
假设在有限次模型修复后，候选模型类固定，且：

\[
\mathcal W_{t+1}\subseteq\mathcal W_t,
\qquad
\mathcal W_t\longrightarrow\mathcal W_\infty
\]
成立于适当集合收敛意义；目标损失关于 \(\omega\) 连续；认证数值上界的求解误差趋于零；存在最终可生成并持续有效的候选结果 \(d^\dagger\) 及 \(\gamma>0\)，满足
\[
\sup_{\omega\in\mathcal W_\infty}
L(d^\dagger,\omega)
\le\varepsilon-\gamma.
\]

若证据获取策略不永久忽略实现上述收敛所需的观测，则存在有限 \(t_0\)，使
\[
\overline E_t(d^\dagger)\le\varepsilon,
\qquad t\ge t_0.
\]

该命题要求观测具有相应可辨识性，并具有足够预算。固定数据的重复计算、重采样次数增加或神经预测方差下降，不单独满足上述条件。

### 12.4 价值近似与规划误差

**命题 4（条件性策略误差界）。**  
设比较类具有至多 \(N\) 个剩余决策轮次，所有可执行动作均被评分，并满足
\[
\left|
\widehat\mu_t(a)-Q_{N-t}^\star(H_t,a)
\right|
\le\eta_t,
\]
\[
0\le\widehat\sigma_t(a)\le\overline\sigma_t.
\]

若动作选择满足
\[
R_t(a_t)
\le
\min_aR_t(a)+\zeta_t,
\]
则相对于同一可执行策略类，
\[
\mathbb E[J(\pi_{\mathrm{AI}})-J(\pi^\star)]
\le
\sum_{t=0}^{N-1}
\left(
2\eta_t
+
\kappa\overline\sigma_t
+
\zeta_t
\right).
\]

其中 \(\eta_t\) 包含表示、世界模型、尾部价值与有限前瞻造成的综合价值误差，\(\zeta_t\) 为搜索误差。

该界不包含未经搜索或不属于比较类的工具程序。

### 12.5 离线求解范围

有限交互图上的精确 Bellman 回代在有限层数内完成，其最优性相对于给定图与转移模型成立。

神经网络训练采用有限迭代与验证集选择，不以训练损失下降替代全局最优性证明。最终估计的统计保证由命题 2 的认证条件给出，规划效率由独立测试任务上的任务损失、成本与比较类遗憾评估。

---

## 13. 最终输出与证明档案

### 13.1 输出状态

系统输出状态为
\[
s_{\mathrm{out}}
\in
\left\{
\begin{aligned}
&\texttt{CERTIFIED\mbox{-}ACT},\\
&\texttt{CERTIFIED\mbox{-}NO\mbox{-}ROUTE},\\
&\texttt{UNRESOLVED},\\
&\texttt{OPERATIONAL}.
\end{aligned}
\right.
\]

\(\texttt{CERTIFIED\mbox{-}ACT}\) 返回具有根证书的最终结果。

\(\texttt{CERTIFIED\mbox{-}NO\mbox{-}ROUTE}\) 返回相对于指定程序类、观测权限与认证目标的无可认证路线证明。

\(\texttt{UNRESOLVED}\) 返回预算、技术、模型或证据不足状态。

\(\texttt{OPERATIONAL}\) 仅在预先登记的运行模式允许时返回未认证数值结果，并将认证字段置空。

### 13.2 根证书

根证书定义为
\[
\mathcal C_{\mathrm{root}}
=
\left(
\mathrm{claim},
\mathrm{scope},
\mathrm{estimand},
\varepsilon,\delta,
\Omega,
\mathcal D_{\mathrm{ref}},
\overline E_T,
\overline B_T,
\mathrm{assumptions},
\mathrm{proof\_dependencies}
\right).
\]

证书登记数据版本、模型版本、有效样本、统计构造、数值求解误差与错误概率支出。

### 13.3 支持证据子图

从完整执行图提取支持根声明的最小依赖闭包：
\[
\mathcal E_{\mathrm{sup}}
=
\operatorname{Ancestors}
(\mathcal C_{\mathrm{root}}).
\]

不支持最终声明的失败分支、探索记录与被替换结果保留在完整账本中，不并入根证书的支持证据。

### 13.4 残余不确定性

对类别比例报告
\[
I_{T,k}
=
\left[
\inf_{\omega\in\mathcal W_T}\beta_k(\omega),
\;
\sup_{\omega\in\mathcal W_T}\beta_k(\omega)
\right],
\qquad k=1,\ldots,K.
\]

预测不确定性、统计置信范围与未解决建模假设分别登记：
\[
\mathcal U_T
=
\left(
\widehat\sigma_{\mathrm{ep}},
\widehat\sigma_{\mathrm{al}},
\{I_{T,k}\}_{k=1}^{K},
\mathrm{unresolved\_assumptions}
\right).
\]

### 13.5 执行与资源档案

完整轨迹记录每次动作的工具版本、参数、输入工件、随机种子、预测评分、实际返回、门控结果、分支变化及成本。

资源档案满足
\[
\mathcal K_T
=
\left(
C_{\mathrm{init}},
\sum_t c_t,
\mathrm{cost\_by\_type},
\sum_j\delta_j,
N_{\mathrm{calls}},
N_{\mathrm{new\_samples}},
\mathrm{termination\_reason}
\right).
\]

最终系统映射为
\[
\boxed{
\mathcal R_u
\xrightarrow{
\text{数据处理、表示、规划、执行、取证、认证与重规划}
}
\left(
s_{\mathrm{out}},
\widehat\beta_u,
\mathcal C_{\mathrm{root}},
\mathcal E_{\mathrm{sup}},
\mathcal B_T,
\tau,
\mathcal U_T,
\mathcal K_T
\right).
}
\]
