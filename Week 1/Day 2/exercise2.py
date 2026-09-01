# Exercise 2: Dictionary Manipulation

person = {
    "name": "Ali",
    "age": 22,
    "email": "ali@example.com",
    "city": "Karachi"
}

print(f"Name: {person['name']}, Age: {person['age']}, Email: {person['email']}, City: {person['city']}")

# Update values
person["age"] = 23
person["email"] = "newemail@example.com"

print(f"\n--- After Updates ---")
print(f"Keys: {list(person.keys())}")
print(f"Values: {list(person.values())}")
print(f"Items: {list(person.items())}")

# Check if key exists
print(f"\nPhone exists: {'phone' in person}")
print(f"Name exists: {'name' in person}")

# Get with default
phone = person.get("phone", "Not provided")
print(f"Phone: {phone}")

# Add new key
person["phone"] = "+923479094814"
print(f"\nAfter adding phone:")
print(f"All info: {person}")
