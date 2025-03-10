import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random

# 下載 GPT2 Tokenizer 和模型
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 移動到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_question(prompt="請生成一個創意問題:"):
    """
    讓 GPT 生成問題
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output = model.generate(input_ids, num_return_sequences=1, temperature=1.0, top_p=0.95)
    question = tokenizer.decode(output[0], skip_special_tokens=True)
    return question

def generate_multiple_answers(question, num_answers=3):
    """
    讓 GPT 針對同一問題生成多個不同答案
    """
    answers = []
    for _ in range(num_answers):
        input_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
        output = model.generate(input_ids, num_return_sequences=1, temperature=1.2, top_p=0.95)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        answers.append(answer)
    return answers

def contrastive_loss(answers):
    """
    計算對比學習損失，鼓勵不同答案之間的多樣性
    """
    loss = 0
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            #  Create tensors with dtype=torch.float32 to enable gradient calculation
            answer_i = torch.tensor(tokenizer.encode(answers[i]), device=device, dtype=torch.float32, requires_grad=True)
            answer_j = torch.tensor(tokenizer.encode(answers[j]), device=device, dtype=torch.float32, requires_grad=True)
            
            # 計算兩個答案的餘弦相似度
            similarity = torch.cosine_similarity(answer_i, answer_j, dim=-1)
            
            # 讓相似度越小越好（鼓勵答案之間的多樣性）
            loss += torch.exp(-similarity).mean()
            
    return loss / (len(answers) * (len(answers) - 1) / 2)  # 平均損失

# 設置優化器
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# 訓練迴圈
num_epochs = 3

for epoch in range(num_epochs):
    print(f"\n===== Epoch {epoch + 1} =====")
    
    # 讓模型自問自答
    question = generate_question()
    print(f"問題: {question}")
    
    answers = generate_multiple_answers(question, num_answers=3)
    for i, ans in enumerate(answers):
        print(f"  答案 {i+1}: {ans}")
    
    # 計算對比損失
    loss = contrastive_loss(answers)
    
    # 反向傳播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")
