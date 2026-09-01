# Exercise 1: List Operations

foods = ["pizza", "burger", "salad", "pasta", "sushi"]
print(f"Original list: {foods}")
print(f"Length: {len(foods)}")
print(f"Pizza exists: {'pizza' in foods}")

foods.append("tacos")
print(f"After adding tacos: {foods}")
print(f"Length: {len(foods)}")

foods.remove("burger")
print(f"After removing burger: {foods}")

# Additional list operations
print(f"\nFirst food: {foods[0]}")
print(f"Last food: {foods[-1]}")
foods.reverse()
print(f"Reversed: {foods}")
