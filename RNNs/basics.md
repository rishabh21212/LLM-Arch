
# **Neural Language Models: A Friendly Guide with Math Examples**

## **1. Introduction: How Computers Predict Words**

Imagine teaching a computer to guess the next word in a sentence, like a super-smart autocomplete. For example, given the phrase:  
**"the cat sat on the ____"**  
A good guess might be **"mat"**—but how does the computer figure that out?  

This is what **Neural Language Models (NLMs)** do. They learn patterns from text and predict the most likely next word. Let’s break it down step by step, first in simple terms, then with the math behind it.

---

## **2. Setting Up the Word Universe (Vocabulary)**
First, we need a small vocabulary of words the model knows:  
**Vocabulary (V):** `["the", "cat", "sat", "on", "mat", "."]`  
(Total words: **6**)  

**Example Sentence:** `"the cat sat on the"`  

### **Why This Matters**  
The model’s job is to predict the next word after seeing `"the cat sat on the"`. Ideally, it should guess **"mat"** (or maybe **"."** if the sentence ends).  

---

## **3. Turning Words into Numbers (One-Hot Encoding)**
Computers don’t understand words—they need numbers. So, we represent each word as a unique **one-hot vector**:  

| Word | One-Hot Encoding |
|------|------------------|
| "the" | `[1, 0, 0, 0, 0, 0]` |
| "cat" | `[0, 1, 0, 0, 0, 0]` |
| "sat" | `[0, 0, 1, 0, 0, 0]` |
| "on"  | `[0, 0, 0, 1, 0, 0]` |
| "mat" | `[0, 0, 0, 0, 1, 0]` |
| "."   | `[0, 0, 0, 0, 0, 1]` |

### **Why One-Hot Encoding?**  
- Each word gets a unique "ID" the computer can process.  
- Helps the model distinguish between words mathematically.  

---

## **4. The Neural Network’s Brain (Parameters)**
The model has a **hidden state** (like short-term memory) and **weights** (rules for processing words).  

### **Key Components:**
1. **Hidden Dimension (`dₕ`)** = 3  
   - Think of this as the model having **3 "thought slots"** to remember context.  
2. **Weight Matrices (Learned from Data):**  
   - **`Wₓₕ`**: Converts input words into hidden states.  
   - **`Wₕₕ`**: Updates the hidden state based on previous memory.  
   - **`Wₕᵧ`**: Converts hidden states into word predictions.  
3. **Biases (`bₕ`, `bᵧ`)**  
   - Adjust predictions to fit patterns better.  

---

## **5. Processing Words Step-by-Step**
The model reads words one by one, updating its hidden state each time.  

### **Initial State (`h₀`):**  
`[0, 0, 0]` (No memory yet.)  

### **For Each Word:**
1. **Combine current word (`xₜ`) + previous hidden state (`hₜ₋₁`).**  
2. **Apply `tanh` (squishes values between -1 and 1).**  
3. **Update hidden state (`hₜ`).**  

#### **Example Calculation (Simplified):**
Suppose after processing `"the cat sat on the"`, the final hidden state is:  
`h₄ = [0.2, -0.1, 0.4]`  

---

## **6. Predicting the Next Word (Softmax)**
The model converts the hidden state into word probabilities using **softmax**:  

### **Final Output (`ŷ`):**  
`[0.02, 0.08, 0.1, 0.15, 0.6, 0.05]`  

This means:  
- **P("the")** = 2%  
- **P("cat")** = 8%  
- **P("sat")** = 10%  
- **P("on")** = 15%  
- **P("mat")** = 60% 🎉 **(Most likely!)**  
- **P(".")** = 5%  

### **Why "mat"?**  
- The model learned that `"on the mat"` is a common phrase.  
- **0.6 (60%)** is the highest probability.  

---

## **7. The Math Behind It (With Examples)**
Now, let’s formalize the steps with equations.

### **A. Hidden State Update**
At each step `t`, the hidden state `hₜ` is computed as:  

\[
hₜ = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
\]

**Where:**  
- `Wₕₕ` = Hidden-to-hidden weights (shape `3×3`)  
- `Wₓₕ` = Input-to-hidden weights (shape `3×6`)  
- `xₜ` = One-hot input word (shape `6×1`)  
- `bₕ` = Hidden bias (shape `3×1`)  

**Example:**  
If `h₃ = [0.1, -0.2, 0.3]` and `x₄ = "the" = [1,0,0,0,0,0]`, then:  
\[
h₄ = \tanh(W_{hh} h₃ + W_{xh} x₄ + b_h)
\]

---

### **B. Output Prediction (Softmax)**
The predicted probabilities are:  

\[
ŷ = \text{softmax}(W_{hy} h_t + b_y)
\]

**Where:**  
- `Wₕᵧ` = Hidden-to-output weights (shape `6×3`)  
- `bᵧ` = Output bias (shape `6×1`)  

**Softmax Example:**  
If `Wₕᵧ h₄ + bᵧ = [1.0, 0.5, -0.2, 0.3, 2.0, -1.0]`, then:  

\[
\text{softmax}([1.0, 0.5, -0.2, 0.3, 2.0, -1.0]) ≈ [0.02, 0.08, 0.1, 0.15, 0.6, 0.05]
\]

—

