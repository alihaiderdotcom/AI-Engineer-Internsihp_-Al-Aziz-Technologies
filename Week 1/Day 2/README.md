# Week 1, Day 2 - Python Data Structures & Control Flow

**Date:** September 2, 2026  
**Duration:** Full day  
**Focus:** Lists, tuples, sets, dictionaries, control flow (if/else, loops)

---

## Review from Day 1

Yesterday we learned:
- Variables and data types (str, int, float, bool)
- Operators (arithmetic, comparison, logical)
- Basic input/output

Today we build on that foundation with **data structures** that let us store and manipulate collections of data.

---

## Topics Covered Today

### 1. Lists - Ordered, Mutable Collections

```python
# Creating lists
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]  # Can mix types

# Accessing elements (0-indexed)
print(fruits[0])      # "apple"
print(fruits[-1])     # "cherry" (last element)

# Modifying lists
fruits[1] = "blueberry"
fruits.append("orange")          # Add to end
fruits.insert(0, "grape")        # Insert at position
fruits.remove("apple")           # Remove by value
popped = fruits.pop()            # Remove and return last

# List operations
print(len(fruits))               # Length
print("apple" in fruits)         # Check membership
print(fruits + numbers)          # Concatenation
```

---

### 2. Tuples - Ordered, Immutable Collections

```python
# Creating tuples
coordinates = (10, 20)
colors = ("red", "green", "blue")
single = (42,)          # Need comma for single element

# Accessing (same as lists)
print(coordinates[0])   # 10

# Tuples are immutable - cannot modify
# coordinates[0] = 15   # ERROR!

# Unpacking tuples
x, y = coordinates
print(x, y)             # 10 20

# Tuple operations
print(len(colors))
print("red" in colors)
```

---

### 3. Dictionaries - Key-Value Pairs

```python
# Creating dictionaries
student = {
    "name": "Ali",
    "age": 22,
    "grade": "A",
    "courses": ["Python", "ML", "AI"]
}

# Accessing values
print(student["name"])           # "Ali"
print(student.get("age"))        # 22 (safer)
print(student.get("email", "N/A"))  # "N/A" (default)

# Modifying dictionaries
student["age"] = 23
student["email"] = "ali@example.com"
student.pop("grade")             # Remove key

# Dictionary operations
print(len(student))              # Number of keys
print("name" in student)         # Check key exists
print(list(student.keys()))      # All keys
print(list(student.values()))    # All values
print(list(student.items()))     # Key-value pairs
```

---

### 4. Sets - Unordered, Unique Collections

```python
# Creating sets
fruits = {"apple", "banana", "cherry", "apple"}
print(fruits)           # {"apple", "banana", "cherry"} - no duplicates!

# Set operations
fruits.add("orange")
fruits.remove("banana")

# Set math
set1 = {1, 2, 3}
set2 = {3, 4, 5}
print(set1 | set2)      # Union: {1, 2, 3, 4, 5}
print(set1 & set2)      # Intersection: {3}
print(set1 - set2)      # Difference: {1, 2}
```

---

### 5. Indexing and Slicing

```python
data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Indexing
print(data[0])          # 0
print(data[3])          # 3
print(data[-1])         # 9 (last)

# Slicing [start:stop:step]
print(data[2:5])        # [2, 3, 4] (stop is exclusive)
print(data[:3])         # [0, 1, 2] (from beginning)
print(data[5:])         # [5, 6, 7, 8, 9] (to end)
print(data[::2])        # [0, 2, 4, 6, 8] (every 2nd)
print(data[::-1])       # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] (reversed)
```

---

### 6. Control Flow - Conditionals

```python
age = 20

# if/elif/else
if age < 13:
    print("Child")
elif age < 18:
    print("Teen")
elif age < 65:
    print("Adult")
else:
    print("Senior")

# Conditional expression (ternary)
status = "Minor" if age < 18 else "Adult"
```

---

### 7. Control Flow - Loops

#### For Loops
```python
# Loop through list
for fruit in ["apple", "banana", "cherry"]:
    print(fruit)

# Loop with index
for i in range(5):      # 0, 1, 2, 3, 4
    print(i)

# Loop through dictionary
student = {"name": "Ali", "age": 22}
for key, value in student.items():
    print(f"{key}: {value}")

# Loop with enumerate
for index, fruit in enumerate(["apple", "banana"]):
    print(f"{index}: {fruit}")
```

#### While Loops
```python
count = 0
while count < 5:
    print(count)
    count += 1

# Loop control
# break - exit loop
# continue - skip to next iteration
for i in range(10):
    if i == 3:
        continue    # Skip 3
    if i == 7:
        break       # Exit at 7
    print(i)        # 0, 1, 2, 4, 5, 6
```

---

### 8. List and Dictionary Comprehensions

```python
# List comprehension [expression for item in iterable]
squares = [x**2 for x in range(5)]          # [0, 1, 4, 9, 16]
evens = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]

# Dictionary comprehension
squares_dict = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, ...}
```

---

### 9. Nested Data Structures

```python
# List of dictionaries
students = [
    {"name": "Ali", "age": 22, "grade": "A"},
    {"name": "Sara", "age": 21, "grade": "B"},
    {"name": "Ahmed", "age": 23, "grade": "A"}
]

# Accessing nested data
print(students[0]["name"])      # "Ali"
print(students[1]["grade"])     # "B"

# Dictionary with lists
courses = {
    "Python": ["Ali", "Sara", "Ahmed"],
    "ML": ["Ali", "Ahmed"],
    "AI": ["Sara"]
}

print(courses["Python"])        # ["Ali", "Sara", "Ahmed"]
```

---

## Exercises

### Exercise 1: List Operations
Create `exercise1.py`:

```python
# TODO: Create a list of 5 favorite foods
# Print the list
# Add a new food
# Remove a food
# Print length and check if "pizza" exists

# Expected Output:
# Original list: ['pizza', 'burger', 'salad', 'pasta', 'sushi']
# Length: 5
# Pizza exists: True
# After adding tacos: ['pizza', 'burger', 'salad', 'pasta', 'sushi', 'tacos']
# Length: 6
# After removing burger: ['pizza', 'salad', 'pasta', 'sushi', 'tacos']
```

**Solution:**
```python
foods = ["pizza", "burger", "salad", "pasta", "sushi"]
print(f"Original list: {foods}")
print(f"Length: {len(foods)}")
print(f"Pizza exists: {'pizza' in foods}")

foods.append("tacos")
print(f"After adding tacos: {foods}")
print(f"Length: {len(foods)}")

foods.remove("burger")
print(f"After removing burger: {foods}")
```

---

### Exercise 2: Dictionary Manipulation
Create `exercise2.py`:

```python
# TODO: Create a dictionary for a person with:
# - name, age, email, city
# Update the age and email
# Print all keys, values, and items
# Check if "phone" exists (it doesn't)

# Expected Output:
# Name: Ali, Age: 22, Email: ali@example.com, City: Karachi
# Keys: ['name', 'age', 'email', 'city']
# Values: ['Ali', 23, 'newemail@example.com', 'Karachi']
# Phone exists: False
```

**Solution:**
```python
person = {
    "name": "Ali",
    "age": 22,
    "email": "ali@example.com",
    "city": "Karachi"
}

print(f"Name: {person['name']}, Age: {person['age']}, Email: {person['email']}, City: {person['city']}")

person["age"] = 23
person["email"] = "newemail@example.com"

print(f"Keys: {list(person.keys())}")
print(f"Values: {list(person.values())}")
print(f"Phone exists: {'phone' in person}")
```

---

### Exercise 3: Conditionals & Loops
Create `exercise3.py`:

```python
# TODO: Grade calculator
# Take marks from user
# Use conditionals to determine grade:
# >= 90: A
# >= 80: B
# >= 70: C
# >= 60: D
# < 60: F
# Display grade and feedback

# Also calculate average of 5 subjects

# Expected Output:
# Subject 1 marks: 85
# Subject 2 marks: 90
# Subject 3 marks: 78
# Subject 4 marks: 88
# Subject 5 marks: 92
# Average: 86.6
# Grade: B
```

**Solution:**
```python
marks = []
for i in range(5):
    mark = int(input(f"Subject {i+1} marks: "))
    marks.append(mark)

average = sum(marks) / len(marks)
print(f"Average: {average}")

if average >= 90:
    grade = "A"
elif average >= 80:
    grade = "B"
elif average >= 70:
    grade = "C"
elif average >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Grade: {grade}")
```

---

### Exercise 4: List Comprehension
Create `exercise4.py`:

```python
# TODO: Use list comprehension to:
# 1. Create a list of squares (1-10)
# 2. Create a list of even numbers (1-20)
# 3. Create a list of uppercase vowels from a word

# Expected Output:
# Squares: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
# Evens: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# Uppercase vowels from "hello world": ['E', 'O', 'O']
```

**Solution:**
```python
# Squares
squares = [x**2 for x in range(1, 11)]
print(f"Squares: {squares}")

# Even numbers
evens = [x for x in range(1, 21) if x % 2 == 0]
print(f"Evens: {evens}")

# Uppercase vowels
word = "hello world"
vowels = [letter.upper() for letter in word if letter.lower() in "aeiou"]
print(f"Uppercase vowels from '{word}': {vowels}")
```

---

### Exercise 5: Nested Data Structures
Create `exercise5.py`:

```python
# TODO: Create a list of 3 students with name, age, courses
# Print information for each student
# Find which student is taking "Python"
# Create a dictionary of course → students enrolled

# Expected Output:
# Student 1: Ali, Age: 22, Courses: ['Python', 'ML', 'AI']
# Student 2: Sara, Age: 21, Courses: ['Python', 'Web Dev']
# Student 3: Ahmed, Age: 23, Courses: ['ML', 'AI']
# Students taking Python: Ali, Sara
# Course enrollment: {'Python': ['Ali', 'Sara'], 'ML': ['Ali', 'Ahmed'], 'AI': ['Ali', 'Ahmed'], 'Web Dev': ['Sara']}
```

**Solution:**
```python
students = [
    {"name": "Ali", "age": 22, "courses": ["Python", "ML", "AI"]},
    {"name": "Sara", "age": 21, "courses": ["Python", "Web Dev"]},
    {"name": "Ahmed", "age": 23, "courses": ["ML", "AI"]}
]

# Print student info
for i, student in enumerate(students, 1):
    print(f"Student {i}: {student['name']}, Age: {student['age']}, Courses: {student['courses']}")

# Find students taking Python
python_students = [s["name"] for s in students if "Python" in s["courses"]]
print(f"Students taking Python: {', '.join(python_students)}")

# Course enrollment
courses_dict = {}
for student in students:
    for course in student["courses"]:
        if course not in courses_dict:
            courses_dict[course] = []
        courses_dict[course].append(student["name"])

print(f"Course enrollment: {courses_dict}")
```

---

## Key Takeaways

1. **Lists** are ordered, mutable - use when order matters and you need to modify
2. **Tuples** are ordered, immutable - use for fixed data that shouldn't change
3. **Dictionaries** store key-value pairs - use to organize related data
4. **Sets** store unique values - use when duplicates don't matter
5. **Slicing** [start:stop:step] is powerful for extracting data
6. **Conditionals** (if/elif/else) make decisions in code
7. **Loops** (for/while) repeat actions on multiple items
8. **Comprehensions** create lists/dicts concisely

---

## Learning Summary (In Your Own Words)

Today was about mastering data structures and control flow - the building blocks of any real Python program. I learned:

- **Data structures** let me organize data efficiently:
  - Lists for ordered, changeable collections
  - Dictionaries for key-value relationships
  - Tuples for immutable data
  - Sets for unique values

- **Control flow** lets programs make decisions and repeat actions:
  - Conditionals (if/else) determine what code runs
  - Loops (for/while) repeat code multiple times
  - Comprehensions make code cleaner and faster

These concepts are absolutely foundational. Every program I write will use loops, conditionals, and data structures. This is where "algorithmic thinking" begins!

---

## What's Next?

Tomorrow we'll learn **functions and modules** - how to organize code into reusable pieces. Then **object-oriented programming** to structure more complex applications.

**Time spent:** ~4-5 hours  
**Completed:** ✓
