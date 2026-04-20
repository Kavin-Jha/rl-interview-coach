# src/problems.py
# Problem bank for the DSA Interview Coach Agent

PROBLEMS = [
    {
        "id": 0,
        "title": "Two Sum",
        "topic": "Arrays & Hashing",
        "difficulty": 1,  # 1=easy, 2=medium, 3=hard
        "description": "Given an array of integers and a target, return indices of two numbers that add up to the target.",
        "hint": "Consider using a hashmap to store complements.",
        "solution_approach": "Use a dictionary to map value to index. For each number, check if target-number exists in the map."
    },
    {
        "id": 1,
        "title": "Best Time to Buy and Sell Stock",
        "topic": "Sliding Window",
        "difficulty": 1,
        "description": "Given prices array, find the max profit from one buy and one sell.",
        "hint": "Track the minimum price seen so far.",
        "solution_approach": "One pass: track min_price and max_profit as you iterate."
    },
    {
        "id": 2,
        "title": "Valid Parentheses",
        "topic": "Stack",
        "difficulty": 1,
        "description": "Given a string of brackets, determine if it is valid.",
        "hint": "Use a stack and a mapping of closing to opening brackets.",
        "solution_approach": "Push opening brackets. On closing bracket, check if top of stack matches."
    },
    {
        "id": 3,
        "title": "Binary Search",
        "topic": "Binary Search",
        "difficulty": 1,
        "description": "Implement binary search on a sorted array.",
        "hint": "Maintain lo and hi pointers.",
        "solution_approach": "While lo <= hi, check mid. Adjust lo or hi based on comparison."
    },
    {
        "id": 4,
        "title": "Reverse Linked List",
        "topic": "Linked Lists",
        "difficulty": 1,
        "description": "Reverse a singly linked list.",
        "hint": "Use prev, curr, next pointers.",
        "solution_approach": "Iteratively reassign next pointers while tracking prev and curr."
    },
    {
        "id": 5,
        "title": "Longest Substring Without Repeating Characters",
        "topic": "Sliding Window",
        "difficulty": 2,
        "description": "Find the length of the longest substring without repeating characters.",
        "hint": "Use a sliding window with a set.",
        "solution_approach": "Expand right pointer, shrink left when duplicate found."
    },
    {
        "id": 6,
        "title": "Number of Islands",
        "topic": "Graphs",
        "difficulty": 2,
        "description": "Count the number of islands in a 2D grid of 1s and 0s.",
        "hint": "Use DFS or BFS from each unvisited land cell.",
        "solution_approach": "Iterate grid, DFS on each unvisited 1, mark visited, increment count."
    },
    {
        "id": 7,
        "title": "Coin Change",
        "topic": "Dynamic Programming",
        "difficulty": 2,
        "description": "Find the minimum number of coins to make up a given amount.",
        "hint": "Build up solutions from smaller amounts.",
        "solution_approach": "DP array where dp[i] = min coins for amount i. Try each coin at each amount."
    },
    {
        "id": 8,
        "title": "Validate Binary Search Tree",
        "topic": "Trees",
        "difficulty": 2,
        "description": "Determine if a binary tree is a valid BST.",
        "hint": "Pass min/max bounds recursively.",
        "solution_approach": "DFS with (node, min_val, max_val). Each node must be within bounds."
    },
    {
        "id": 9,
        "title": "Merge K Sorted Lists",
        "topic": "Linked Lists",
        "difficulty": 3,
        "description": "Merge k sorted linked lists into one sorted list.",
        "hint": "Use a min-heap to always extract the smallest element.",
        "solution_approach": "Push head of each list into heap. Pop min, push its next, repeat."
    },
    {
        "id": 10,
        "title": "Word Search II",
        "topic": "Graphs",
        "difficulty": 3,
        "description": "Find all words from a list that exist in a 2D board.",
        "hint": "Build a Trie from the word list, then DFS on the board.",
        "solution_approach": "Trie + DFS with backtracking. Prune branches not in Trie."
    },
    {
        "id": 11,
        "title": "Longest Increasing Subsequence",
        "topic": "Dynamic Programming",
        "difficulty": 2,
        "description": "Find the length of the longest strictly increasing subsequence.",
        "hint": "For each element, find the longest subsequence ending at it.",
        "solution_approach": "dp[i] = max(dp[j]+1) for all j < i where nums[j] < nums[i]. O(n^2) or O(n log n)."
    },
]

def get_problems_by_difficulty(difficulty: int):
    return [p for p in PROBLEMS if p["difficulty"] == difficulty]

def get_problems_by_topic(topic: str):
    return [p for p in PROBLEMS if p["topic"] == topic]

def get_all_problem_ids():
    return [p["id"] for p in PROBLEMS]

def get_problem_by_id(problem_id: int):
    for p in PROBLEMS:
        if p["id"] == problem_id:
            return p
    return None