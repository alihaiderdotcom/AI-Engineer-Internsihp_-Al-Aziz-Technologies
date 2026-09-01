# Exercise 4: List Comprehension

# Create squares (1-10)
squares = [x**2 for x in range(1, 11)]
print(f"Squares (1-10): {squares}")

# Create even numbers (1-20)
evens = [x for x in range(1, 21) if x % 2 == 0]
print(f"Even numbers (1-20): {evens}")

# Create odd numbers (1-20)
odds = [x for x in range(1, 21) if x % 2 != 0]
print(f"Odd numbers (1-20): {odds}")

# Uppercase vowels from a word
word = "hello world"
vowels = [letter.upper() for letter in word if letter.lower() in "aeiou"]
print(f"Uppercase vowels from '{word}': {vowels}")

# Dictionary comprehension - word length mapping
words = ["python", "ai", "machine", "learning"]
word_lengths = {word: len(word) for word in words}
print(f"\nWord lengths: {word_lengths}")

# Squares in a dictionary
squares_dict = {x: x**2 for x in range(1, 6)}
print(f"Squares dict: {squares_dict}")

# List of numbers, each multiplied by 2
numbers = [1, 2, 3, 4, 5]
doubled = [x * 2 for x in numbers]
print(f"\nDoubled numbers: {doubled}")
