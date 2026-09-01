# Week 1, Day 1 - Python Fundamentals

**Date:** September 1, 2026  
**Duration:** Full day  
**Focus:** Python basics, syntax, variables, data types, operators, I/O

---

## What is Python?

Python is a high-level, interpreted programming language known for:
- **Simple, readable syntax** - Easy to learn and understand
- **Versatile** - Web, Data Science, AI/ML, Automation, Scientific Computing
- **Rich ecosystem** - NumPy, Pandas, scikit-learn, TensorFlow, PyTorch
- **Industry standard** - Used by tech giants: Google, Facebook, Netflix, NASA

### Why Python for AI?
- Dominant language in ML/AI community
- Extensive AI/ML libraries
- Fast prototyping capabilities
- Strong community support

---

## Topics Covered Today

### 1. Python Installation & Environment Setup

Check Python version:
```bash
python3 --version
```

### 2. Python Syntax Basics

```python
# Comments start with #
"""
Multi-line comments use triple quotes
"""
```

### 3. Variables and Data Types

```python
# Variables - no need to declare type
name = "Ali"           # String
age = 22               # Integer
height = 5.9           # Float
is_student = True      # Boolean

# Check type
print(type(name))      # <class 'str'>
print(type(age))       # <class 'int'>
```

### 4. Operators

```python
# Arithmetic
x = 10
y = 3
print(x + y)           # 13
print(x - y)           # 7
print(x * y)           # 30
print(x / y)           # 3.33
print(x // y)          # 3 (floor division)
print(x % y)           # 1 (modulo)
print(x ** y)          # 1000 (exponent)

# Comparison
print(x > y)           # True
print(x == y)          # False
print(x != y)          # True

# Logical
print(True and False)  # False
print(True or False)   # True
print(not True)        # False
```

### 5. Input and Output

```python
# Output
print("Hello World!")
print("Name:", name)

# Input
user_name = input("Enter your name: ")
print("Hello, " + user_name)
```

### 6. Type Conversion

```python
num_str = "42"
num_int = int(num_str)        # String to int
num_float = float(num_str)    # String to float
back_to_str = str(42)         # Int to string
```

### 7. Basic Error Handling

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
```

---

## Exercises

### Exercise 1: Variable Assignment & Output
Create a file `exercise1.py`:

```python
# TODO: Create variables for:
# - Your name (string)
# - Your age (integer)
# - Your height in meters (float)
# - Whether you're a student (boolean)
# Then print all of them

# Expected Output:
# Name: Ali
# Age: 22
# Height: 5.9
# Student: True
```

**Solution:**
```python
name = "Ali"
age = 22
height = 5.9
is_student = True

print("Name:", name)
print("Age:", age)
print("Height:", height)
print("Student:", is_student)
```

---

### Exercise 2: Arithmetic Operations
Create a file `exercise2.py`:

```python
# TODO: Take two numbers as input from user
# Perform all arithmetic operations (+, -, *, /, //, %, **)
# Display results in a formatted way

# Expected Output:
# Enter first number: 15
# Enter second number: 4
# 15 + 4 = 19
# 15 - 4 = 11
# 15 * 4 = 60
# 15 / 4 = 3.75
# 15 // 4 = 3
# 15 % 4 = 3
# 15 ** 4 = 50625
```

**Solution:**
```python
num1 = int(input("Enter first number: "))
num2 = int(input("Enter second number: "))

print(f"{num1} + {num2} = {num1 + num2}")
print(f"{num1} - {num2} = {num1 - num2}")
print(f"{num1} * {num2} = {num1 * num2}")
print(f"{num1} / {num2} = {num1 / num2}")
print(f"{num1} // {num2} = {num1 // num2}")
print(f"{num1} % {num2} = {num1 % num2}")
print(f"{num1} ** {num2} = {num1 ** num2}")
```

---

### Exercise 3: Temperature Converter
Create a file `exercise3.py`:

```python
# TODO: Convert temperature from Celsius to Fahrenheit
# Formula: F = (C * 9/5) + 32
# Take celsius as input and display fahrenheit

# Expected Output:
# Enter temperature in Celsius: 25
# 25°C = 77.0°F
```

**Solution:**
```python
celsius = float(input("Enter temperature in Celsius: "))
fahrenheit = (celsius * 9/5) + 32
print(f"{celsius}°C = {fahrenheit}°F")
```

---

## Key Takeaways

1. **Variables** store data; Python figures out the type automatically
2. **Data types** include strings, integers, floats, booleans
3. **Operators** perform calculations and comparisons
4. **Input/Output** lets programs interact with users
5. **Type conversion** allows changing between data types
6. **Error handling** makes programs more robust

---

## Learning Summary (In Your Own Words)

Today I learned the absolute basics of Python - how to write simple programs that:
- Store and manipulate data in variables
- Perform arithmetic and logical operations
- Accept user input and display output
- Handle basic errors gracefully

The most important concept is understanding Python's simple syntax and the different data types. Everything in AI/ML programming builds on these fundamentals!

---

## Next Steps

Tomorrow we'll explore **data structures** (lists, dictionaries, etc.) and **control flow** (loops, conditionals) to handle more complex problems.

**Time spent:** ~3-4 hours  
**Completed:** ✓
