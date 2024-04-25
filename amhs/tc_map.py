import hashlib
#hash: SHA-256
def hash_generator(input_string):
    hash_object = hashlib.sha256()
    hash_object.update(input_string.encode('utf-8'))
    hex_digest = hash_object.hexdigest()
    return hex_digest

if __name__ == "__main__":
    user_input = input("Enter a string to generate its SHA-256 hash: ")
    hashed_value = hash_generator(user_input)
    print(f"SHA-256 hash of '{user_input}': {hashed_value}")

'''
   CREATE TABLE my_table (
     id INT AUTO_INCREMENT PRIMARY KEY,
     long_column VARCHAR(255),
     -- 其他列...
     INDEX idx_long_column (long_column) USING HASH
   ) ENGINE=MEMORY;

#    插入数据
    INSERT INTO my_table (long_column, ...) VALUES ('some_value', ...);

# 索引数据
   SELECT * FROM my_table WHERE long_column = 'specific_value';
'''