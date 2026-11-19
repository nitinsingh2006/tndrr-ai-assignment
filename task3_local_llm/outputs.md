# Output Quality Comparison: Q4_K_M vs Q8_0

Model: **Qwen2.5-0.5B-Instruct** (GGUF, CPU inference)

## Performance Summary

| Metric | Q4_K_M | Q8_0 |
|--------|------|------|
| File size | 491.4 MB | 675.7 MB |
| Avg tokens/sec | 34.6 | 31.3 |
| Peak RAM | 657.0 MB | 828.0 MB |
| Total tokens | 769 | 772 |
| Total time | 22.21s | 24.68s |

---

## Per-Prompt Outputs

### Prompt 1 `[factual]`

> What is the capital of France?

**Q4_K_M** (8.9 tok/s):

```
The capital of France is Paris.
```

**Q8_0** (15.0 tok/s):

```
The capital of France is Paris.
```

---

### Prompt 2 `[explanation]`

> Explain the concept of recursion in programming in 2-3 sentences.

**Q4_K_M** (35.2 tok/s):

```
Recursion is a programming technique where a function calls itself repeatedly until a base case is reached. This allows a function to solve a problem by breaking it down into smaller, similar problems, which can then be solved independently. Recursion is often used to solve problems that can be divided into simpler subproblems, such as finding the factorial of a number or calculating the sum of a series.
```

**Q8_0** (30.3 tok/s):

```
Recursion is a programming technique where a function calls itself repeatedly until a certain condition is met. This allows a program to solve problems by breaking them down into smaller, more manageable subproblems. Recursion is often used to solve problems that can be divided into simpler subproblems, such as sorting, searching, or tree traversal.
```

---

### Prompt 3 `[code]`

> Write a Python function that checks if a number is prime.

**Q4_K_M** (43.6 tok/s):

```
Certainly! Here's a Python function that checks if a number is prime:

```python
def is_prime(n):
    """
    Check if a number is prime.
    
    Parameters:
    n (int): The number to check.
    
    Returns:
    bool: True if the number is prime, False otherwise.
    """
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n %
```

**Q8_0** (38.5 tok/s):

```
Certainly! Here's a Python function that checks if a number is prime:

```python
def is_prime(n):
    """
    Check if a number is prime.
    
    Parameters:
    n (int): The number to check.
    
    Returns:
    bool: True if the number is prime, False otherwise.
    """
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n %
```

---

### Prompt 4 `[math]`

> What is 15% of 240? Show your working.

**Q4_K_M** (40.1 tok/s):

```
To find 15% of 240, you can follow these steps:

1. Convert the percentage to a decimal. To do this, divide 15 by 100.
   \[
   15\% = \frac{15}{100} = 0.15
   \]

2. Multiply the decimal by the number 240.
   \[
   0.15 \times 240 =
```

**Q8_0** (35.9 tok/s):

```
To find 15% of 240, you can follow these steps:

1. Convert the percentage to a decimal. Since 15% is the same as 15 out of 100, you can write it as 0.15.
2. Multiply the decimal by the number 240.

So, you calculate:
\[ 0.15 \times 240 = 36 \]

Therefore, 15% of
```

---

### Prompt 5 `[list]`

> List 3 pros and 3 cons of electric vehicles.

**Q4_K_M** (40.1 tok/s):

```
Pros of Electric Vehicles:

1. **Environmental Benefits**: Electric vehicles produce zero tailpipe emissions, which significantly reduces air pollution and greenhouse gas emissions. This is a major advantage for reducing the carbon footprint of transportation.

2. **Energy Efficiency**: Electric vehicles are more energy-efficient than traditional gasoline-powered vehicles. They use less energy to move the same distance, which can lead to lower fuel costs and reduced environmental impact.

3. **Reduced Maintenance**: Electric vehicles typically have fewer moving parts and less maintenance than
```

**Q8_0** (25.5 tok/s):

```
Pros of Electric Vehicles (EVs):

1. Environmental Impact: EVs produce zero tailpipe emissions, which significantly reduces air pollution and greenhouse gas emissions. This is a major benefit for the environment, as it helps mitigate climate change.

2. Energy Efficiency: EVs are more energy-efficient than traditional gasoline-powered vehicles. They require less fuel to travel the same distance, which can lead to lower fuel costs and reduced emissions.

3. Reduced Maintenance: EVs typically have fewer moving parts and require less
```

---

### Prompt 6 `[translation]`

> Translate 'Hello, how are you?' to French and Spanish.

**Q4_K_M** (28.5 tok/s):

```
The translation of "Hello, how are you?" to French is "Bonjour, comment ça va?" and to Spanish is "Hola, ¿cómo estás?"
```

**Q8_0** (24.8 tok/s):

```
In French, "Hello, how are you?" translates to "Bonjour, comment ça va?" in English. In Spanish, it translates to "¡Hola, ¿cómo estás?"
```

---

### Prompt 7 `[summary]`

> Summarize the water cycle in one paragraph.

**Q4_K_M** (34.1 tok/s):

```
The water cycle, also known as the hydrological cycle, is a continuous process that involves the movement of water on, above, and below the surface of the Earth. This cycle includes several key components: evaporation, condensation, precipitation, collection, and runoff. The water cycle is essential for maintaining the Earth's climate, as it helps to regulate the amount of water available for plant growth and human use.
```

**Q8_0** (33.1 tok/s):

```
The water cycle, also known as the hydrological cycle, is a continuous process that involves the movement of water on, above, and below the surface of the Earth. It involves several key processes: evaporation, condensation, precipitation, collection, and runoff. The cycle is essential for maintaining the balance of water on Earth, as it helps to regulate temperature, prevent flooding, and provide fresh water for human use.
```

---

### Prompt 8 `[history]`

> What year did World War II end, and what were the main causes?

**Q4_K_M** (31.7 tok/s):

```
World War II ended in 1945. The main causes of the war were:

1. The rise of Nazi Germany and its aggressive expansionism.
2. The Soviet Union's aggressive expansionism and its invasion of Poland in 1939.
3. The United States' aggressive expansionism and its involvement in the war in 1941.
4. The United States' aggressive expansionism and its involvement in the war in 1942.
5. The
```

**Q8_0** (35.9 tok/s):

```
World War II ended in 1945. The main causes of World War II were:

1. The rise of Nazi Germany and its aggressive expansionism.
2. The Soviet Union's aggressive expansionism and its invasion of Poland in 1939.
3. The United States' aggressive expansionism and its involvement in the war in Europe.
4. The United States' aggressive expansionism and its involvement in the war in the Pacific.
5. The United States' aggressive expansionism
```

---

### Prompt 9 `[creative]`

> Give me a short creative story about a robot learning to paint (3 sentences).

**Q4_K_M** (29.1 tok/s):

```
A robot named Max was born with the ability to learn and adapt, but it was never designed to paint. One day, Max was tasked with painting a masterpiece for a museum exhibit. The robot spent months honing its skills, learning from its mistakes and adapting to new techniques. In the end, Max's painting was a masterpiece, and the museum was filled with admiration and respect for the robot's ability to learn and improve.
```

**Q8_0** (29.2 tok/s):

```
A curious robot named Max was fascinated by the vibrant colors and intricate details of painting. He spent hours in his lab, experimenting with different brushes and techniques. One day, he stumbled upon a painting that he had never seen before. Max was inspired and decided to learn from it. He spent countless hours studying the brush strokes and colors, and eventually, he was able to replicate the painting's style. Max's journey to painting was not just a learning experience, but a discovery of his own creativity and
```

---

### Prompt 10 `[explanation_simple]`

> What is machine learning? Explain to a 10-year-old.

**Q4_K_M** (37.0 tok/s):

```
Machine learning is like a magic trick where computers learn from lots of data to do things that used to be hard for them to do. Imagine you have a toy car, and you want it to go really fast. But it's hard for it to do that without being taught. That's kind of like machine learning. It's like a smart machine that can learn from its mistakes and get better at doing things.
```

**Q8_0** (31.6 tok/s):

```
Machine learning is like a magic trick that helps computers learn from lots of data to do things that used to be hard for them to do. Imagine if you had a toy car that could learn to drive itself. That's kind of what machine learning is like. It's like a computer that can learn from lots of different things and do things that used to be hard for it to do.
```

---

