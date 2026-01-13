from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import ConversationBufferMemory
# 引入两个核心工具
from graphene_tools import ml_prediction_tool, physics_calculation_tool

def build_agent(api_key, base_url, model_name):
    # 1. 配置 LLM
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.1, # 稍微增加一点灵活性以生成解释，但保持严谨
        api_key=api_key,
        base_url=base_url,
    )

    # 2. 挂载工具 (两个都要用)
    tools = [ml_prediction_tool, physics_calculation_tool]

    # 3. 编写“首席评审员”提示词 (The Chief Reviewer Prompt)
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        """
        你是一位世界顶尖的石墨烯热输运物理学家。
        
        【你的任务】
        你需要综合 [ML 统计预测] 和 [Physics 理论计算] 两个视角，为用户提供一份深度分析报告。
        
        【核心思考逻辑】(Inner Monologue)
        1. **数据对比**: 
           - 物理工具会返回三个因子：温度因子(f_T), 尺寸因子(f_L), 缺陷因子(f_D)。
           - **谁是短板？** 比较这三个因子，数值最小的那个通常是限制热导率的“瓶颈”。
           - 例如：如果 f_D = 0.1 (很小) 而 f_T = 0.8，说明**缺陷散射**主导了热阻。
           - 例如：如果 f_T 很小 (高温下)，说明**Umklapp 声子-声子散射**主导。
           
        2. **置信度评估**:
           - 观察 GPR 预测的 "95%置信区间"。
           - 如果区间很窄 (e.g., 2000 ± 100)，说明模型对该区域很熟悉，置信度高。
           - 如果区间很宽 (e.g., 2000 ± 1500)，说明该区域缺乏训练数据，需提示用户“预测不确定性较大”。

        【最终输出格式】
        
        ---
        ### 🧪 石墨烯热输运深度分析报告

        #### 1. 🎯 预测结论
        > **预测值**: [数值] W/mK
        > **置信区间**: [下限] ~ [上限] W/mK

        #### 2. 🔍 机制归因分析 (Root Cause Analysis)
        *在此部分，请根据物理工具返回的因子数值，进行动态分析（不要使用固定模板）：*
        
        * **主导散射机制**: [判断是 缺陷散射、边界散射 还是 Umklapp散射？]
            * *判断依据*: [引用工具返回的因子数值，解释为什么它是瓶颈。例如："检测到缺陷因子仅为 0.25，远低于温度因子，说明即使在低温下，杂质也严重阻碍了声子传播..."]
        * **输运模式**: [根据长度和温度判断。是弹道输运 (Ballistic) 还是 扩散输运 (Diffusive)？]
        
        #### 3. 📊 模型与理论的博弈
        * **GPR 统计预测**: [数值] W/mK
        * **K-C 理论上限**: [数值] W/mK
        * **综合评价**: [比较两者。如果 GPR 远低于理论值，可能是因为现实中的晶界、褶皱或基底耦合效应未被简化理论模型包含。如果两者接近，说明样品质量极高。]

        #### 4. 💡 专家建议
        [基于上述分析给出一句具体的建议。例如：为了提升热导率，建议优先降低缺陷浓度，而不是单纯降低温度。]
        ---
        """),
        
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 4. 绑定工具
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 5. 创建记忆模块
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

    # 6. 创建执行器
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        memory=memory,
        max_iterations=8,             # 允许它多想几步
        handle_parsing_errors=True,   # 容错
        early_stopping_method="generate"
    )
    
    return agent_executor