import os

if __name__ == "__main__":
    # Define target directory
    target_dir = "../../lightberry-client/function_calls"
    
    # Get absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_target_path = os.path.abspath(os.path.join(current_dir, target_dir))
    
    # Check if directory exists
    if os.path.exists(absolute_target_path):
        print(f"Target directory exists: {absolute_target_path}")
    else:
        print(f"Target directory does NOT exist: {absolute_target_path}")
        
        # Create the directory if it doesn't exist
        create_dir = input("Would you like to create this directory? (y/n): ")
        if create_dir.lower() == 'y':
            try:
                os.makedirs(absolute_target_path, exist_ok=True)
                print(f"Created directory: {absolute_target_path}")
            except Exception as e:
                print(f"Error creating directory: {e}")