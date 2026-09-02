# Week 1, Day 3 - Functions, Modules & Object-Oriented Programming

**Date:** September 3, 2026  
**Focus:** Reusable functions, modules, packages, virtual environments, pip, and introductory OOP

## Learning Objectives

By the end of today, I can:

- Define functions with parameters and return values.
- Use default parameters, `*args`, `**kwargs`, and lambda functions.
- Explain local and global scope.
- Split code into reusable modules and import them.
- Explain packages, virtual environments, `pip`, and `requirements.txt`.
- Create classes with constructors and methods.
- Use basic inheritance to extend a class.

## Core Concepts

### Functions

```python
def calculate_average(values, precision=2):
    if not values:
        return 0
    return round(sum(values) / len(values), precision)
```

A function should generally perform one clear task. Parameters provide input, and `return` provides a result to the caller.

```python
def describe_student(name, *subjects, **details):
    return {
        "name": name,
        "subjects": list(subjects),
        "details": details,
    }
```

- `*args` collects extra positional arguments into a tuple.
- `**kwargs` collects extra named arguments into a dictionary.
- A lambda is a short anonymous function, such as `lambda score: score >= 50`.

### Scope

Variables created inside a function are local to that function. A function can read a global value, but passing values as parameters is usually clearer and easier to test.

### Modules and Packages

A module is a Python file that contains reusable code. Import only the names needed by the current file:

```python
from calculator import calculate_average, get_grade
```

A package is a directory containing related modules. In larger projects, packages help separate responsibilities and keep files manageable.

### Virtual Environments and pip

Create and activate an isolated environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r "Week 1/Day 3/requirements.txt"
```

A `requirements.txt` file records the packages needed by a project. This Day 3 exercise uses only the Python standard library, so no external dependency is required yet.

### Classes and Objects

A class defines behavior and data. An object is an instance of that class.

```python
class Student:
    def __init__(self, name, scores):
        self.name = name
        self.scores = scores

    def average(self):
        return sum(self.scores) / len(self.scores)
```

Inheritance allows a specialized class to reuse and extend a base class:

```python
class Intern(Student):
    def introduce(self):
        return f"I am {self.name}, an AI Engineering intern."
```

## Practical Exercise

Run the application from this folder:

```bash
python3 main.py
```

The program imports functions from `calculator.py`, creates an `Intern` object from `student.py`, calculates an average and grade, and prints the student's profile.

Expected output includes:

```text
Student: Ali Haider
Average: 86.60
Grade: B
Role: AI Engineering Intern
Subjects: Python, Machine Learning, AI
```

## Files

- `calculator.py` - reusable functions and a simple `*args/**kwargs` example
- `student.py` - `Student` base class and `Intern` subclass
- `main.py` - application entry point using imports and objects
- `requirements.txt` - project dependency record

## Learning Summary

Today I learned how to turn a long script into smaller, reusable pieces. Functions keep repeated logic in one place, modules separate responsibilities, and classes group related data with the operations that use it. Virtual environments and `requirements.txt` make a project's Python setup reproducible.

## Completion Checklist

- [x] Functions with parameters and return values
- [x] Default parameters, `*args`, and `**kwargs`
- [x] Lambda functions and scope
- [x] Modules and imports
- [x] Virtual environment and `pip` concepts
- [x] Classes, constructors, and methods
- [x] Basic inheritance
- [x] Working OOP application
