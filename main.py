import math
from typing import TypeVar, Generator, List, Tuple, Optional
from collections import deque
import json
from queue import SimpleQueue

T = TypeVar("T")  # represents generic type
# represents a Node object (forward-declare to use in Node __init__)
Node = TypeVar("Node")
# represents a custom type used in application
AVLWrappedDictionary = TypeVar("AVLWrappedDictionary")


class Node:
    """
    Implementation of an BST and AVL tree node.
    Do not modify.
    """
    # preallocate storage
    __slots__ = ["value", "parent", "left", "right", "height"]

    def __init__(self, value: T, parent: Node = None,
                 left: Node = None, right: Node = None) -> None:
        """
        Construct an AVL tree node.

        :param value: value held by the node object
        :param parent: ref to parent node of which this node is a child
        :param left: ref to left child node of this node
        :param right: ref to right child node of this node
        """
        self.value = value
        self.parent, self.left, self.right = parent, left, right
        self.height = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"

    def __str__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return repr(self)


####################################################################################################

class BinarySearchTree:
    """
    Implementation of an BSTree.
    Modify only below indicated line.
    """

    # preallocate storage
    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty BST tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the BSTree as a string.

        :return: string representation of the BST tree
        """
        if self.origin is None:
            return "Empty BST Tree"

        lines = pretty_print_binary_tree(self.origin, 0, False, '-')[0]
        return "\n" + "\n".join((line.rstrip() for line in lines))

    def __str__(self) -> str:
        """
        Represent the BSTree as a string.

        :return: string representation of the BSTree
        """
        return repr(self)

    def visualize(self, filename="bst_visualization.svg"):
        """
        Generates an svg image file of the binary tree.

        :param filename: The filename for the generated svg file. Should end with .svg.
        Defaults to output.svg
        """
        svg_string = svg(self.origin, node_radius=20)
        print(svg_string, file=open(filename, 'w'))
        return svg_string

    ########################################
    # My Code Below #
    ########################################

    def height(self, root: Node) -> int:
        """
        Calculates and returns the height of a subtree in the BSTree. Returns -1 for an empty subtree.
        Time/Space: O(1)/O(1).

        :param root: Node: The root of the subtree.
        :return: Height of the subtree at root, or -1 if root is None.
        """
        if root is None:
            return -1
        return root.height

    def insert(self, root: Node, val: T) -> None:
        """
        Inserts a node with the value val into the subtree rooted at root. Updates size and origin attributes.
        Time/Space: O(h)/O(1), where h is the height of the tree.

        :param root: Node: The root of the subtree.
        :param val: T: The value to insert.
        :return: None.
        """

        if root is None:
            self.size += 1
            new_node = Node(val)
            if self.origin is None:
                self.origin = new_node
            return new_node

        elif val < root.value:
            root.left = self.insert(root.left, val)
            if root.left.parent is None:  # set parent
                root.left.parent = root

        elif val > root.value:
            root.right = self.insert(root.right, val)
            if root.right.parent is None:  # set parent
                root.right.parent = root
        else:
            return root  # value already exists in the tree, do nothing

        # update height of the node
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        return root


    def remove(self, root: Node, val: T) -> Optional[Node]:
        """
        Removes the node with value val from the subtree rooted at root. Updates size, origin, and structure.
        Time/Space: O(h)/O(1), where h is the height of the tree.

        :param root: Node: The root of the subtree.
        :param val: T: The value to be deleted.
        :return: The root of the new subtree after removal.
        """

        if root is None:
            return root

        if val < root.value:
            root.left = self.remove(root.left, val)
        elif val > root.value:
            root.right = self.remove(root.right, val)
        else:
            if root.left is None or root.right is None:
                self.size -= 1
                temp = root.left if root.left else root.right
                if temp is None:
                    root = None
                else:
                    root = temp
            else:
                temp = root.left
                while temp.right:
                    temp = temp.right
                root.value = temp.value
                root.left = self.remove(root.left, temp.value)

        if root:
            root.height = 1 + max(self.height(root.left), self.height(root.right))

        return root

    def search(self, root: Node, val: T) -> Optional[Node]:
        """
        Searches for and returns the Node containing value val in the subtree rooted at root.
        If val is not present, returns the Node below which val would be inserted as a child.
        Time/Space: O(h)/O(1), where h is the height of the tree.

        :param root: Node: The root of the subtree.
        :param val: T: The value to search for.
        :return: Node containing val, or Node below which val would be inserted if not present.
        """

        if root is None or root.value == val:
            return root

        if val < root.value:
            if root.left is None:
                return root
            return self.search(root.left, val)
        else:
            if root.right is None:
                return root
            return self.search(root.right, val)


class AVLTree:
    """
    Implementation of an AVL tree.
    Modify only below indicated line.
    """

    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty AVL tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree as a string.

        :return: string representation of the AVL tree
        """
        if self.origin is None:
            return "Empty AVL Tree"

        return super(AVLTree, self).__repr__()

    def __str__(self) -> str:
        """
        Represent the AVLTree as a string.

        :return: string representation of the BSTree
        """
        return repr(self)

    def visualize(self, filename="avl_tree_visualization.svg"):
        """
        Generates an svg image file of the binary tree.

        :param filename: The filename for the generated svg file. Should end with .svg.
        Defaults to output.svg
        """
        svg_string = svg(self.origin, node_radius=20)
        print(svg_string, file=open(filename, 'w'))
        return svg_string

    ########################################
    # Implement functions below this line. #
    ########################################

    def height(self, root: Node) -> int:
        """
        Calculates the height of a subtree in the AVL tree, handling cases where root might be None.
        The height of an empty subtree is -1.
        Parameters: root (Node): The root node of the subtree.
        Returns: Height of the subtree rooted at root.
        Time/Space: O(1)/O(1)
        """

        if root is None:
            return -1
        return root.height

    def left_rotate(self, root: Node) -> Optional[Node]:
        """
        Performs a left rotation on the subtree rooted at root, returning the new root.
        Parameters: root (Node): The root node of the subtree.
        Returns: The root of the new subtree post-rotation.
        Time/Space: O(1)/O(1)
        """

        if root is None or root.right is None:
            return root

        new_root = root.right
        root.right = new_root.left
        if new_root.left:
            new_root.left.parent = root

        new_root.parent = root.parent
        if root.parent is None:
            self.origin = new_root
        elif root == root.parent.left:
            root.parent.left = new_root
        else:
            root.parent.right = new_root

        new_root.left = root
        root.parent = new_root

        # update heights
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + max(self.height(new_root.left), self.height(new_root.right))

        return new_root

    def right_rotate(self, root: Node) -> Optional[Node]:
        """
        Performs a right rotation on the subtree rooted at root, returning the new root.
        Parameters: root (Node): The root node of the subtree.
        Returns: The root of the new subtree post-rotation.
        Time/Space: O(1)/O(1)
        """

        if root is None or root.left is None:
            return root

        new_root = root.left
        root.left = new_root.right
        if new_root.right:
            new_root.right.parent = root

        new_root.parent = root.parent
        if root.parent is None:
            self.origin = new_root
        elif root == root.parent.left:
            root.parent.left = new_root
        else:
            root.parent.right = new_root

        new_root.right = root
        root.parent = new_root

        # update heights
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + max(self.height(new_root.left), self.height(new_root.right))

        return new_root

    def balance_factor(self, root: Node) -> int:
        """
        Computes the balance factor of the subtree rooted at root.
        Parameters: root (Node): The root node of the subtree.
        Returns: An integer representing the balance factor of root.
        Time/Space: O(1)/O(1)
        """

        if root is None:
            return 0

        return self.height(root.left) - self.height(root.right)

    def rebalance(self, root: Node) -> Optional[Node]:
        """
        Rebalances the subtree rooted at root if it is unbalanced, returning the new root.
        Parameters: root (Node): The root of the subtree.
        Returns: The root of the new, potentially rebalanced subtree.
        Time/Space: O(1)/O(1)
        """

        if root is None:
            return root

        balance = self.balance_factor(root)

        # left-Heavy
        if balance > 1:
            # left-right case
            if self.balance_factor(root.left) < 0:
                root.left = self.left_rotate(root.left)
            # Left-Left Case
            return self.right_rotate(root)

        # right-heavy
        if balance < -1:
            # right-left case
            if self.balance_factor(root.right) > 0:
                root.right = self.right_rotate(root.right)
            # right-right case
            return self.left_rotate(root)

        return root

    def insert(self, root: Node, val: T) -> Optional[Node]:
        """
        Inserts a new node with value val into the subtree rooted at root, balancing the subtree as necessary.
        Returns: The root of the new, balanced subtree.
        Time/Space: O(log n)/O(1)
        """

        # base case: inserting into an empty subtree
        if root is None:
            self.size += 1
            new_node = Node(val)
            if self.origin is None:
                self.origin = new_node
            return new_node

        # recursive insertion
        if val < root.value:
            root.left = self.insert(root.left, val)
            if root.left.parent is None:
                root.left.parent = root
        elif val > root.value:
            root.right = self.insert(root.right, val)
            if root.right.parent is None:
                root.right.parent = root
        else:
            return root  # value already exists in the tree, do nothing

        # update the height of the node
        root.height = 1 + max(self.height(root.left), self.height(root.right))

        # rebalance the tree
        return self.rebalance(root)

    def remove(self, root: Node, val: T) -> Optional[Node]:
        """
        Removes the node with value val from the subtree rooted at root, balancing the subtree as necessary.
        Returns: The root of the new, balanced subtree.
        Time/Space: O(log n)/O(1)
        """

        if root is None:
            return root

        if val < root.value:
            root.left = self.remove(root.left, val)
        elif val > root.value:
            root.right = self.remove(root.right, val)
        else:
            if root.left is None or root.right is None:
                self.size -= 1
                temp = root.left if root.left else root.right
                if temp is None:
                    root = None
                else:
                    root = temp
            else:
                temp = root.left
                while temp.right:
                    temp = temp.right
                root.value = temp.value
                root.left = self.remove(root.left, temp.value)

        if root:
            root.height = 1 + max(self.height(root.left), self.height(root.right))
            root = self.rebalance(root)

        return root

    def min(self, root: Node) -> Optional[Node]:
        """
        Returns the Node containing the smallest value within the subtree rooted at root.
        Returns: Node with the smallest value in the subtree.
        Time/Space: O(log n)/O(1)
        """

        if root is None:
            return None

        while root.left:
            root = root.left

        return root

    def max(self, root: Node) -> Optional[Node]:
        """
        Returns the Node containing the largest value within the subtree rooted at root.
        Returns: Node with the largest value in the subtree.
        Time/Space: O(log n)/O(1)
        """

        if root is None:
            return None

        while root.right:
            root = root.right

        return root

    def search(self, root: Node, val: T) -> Optional[Node]:
        """
        Searches for the Node with the value val within the subtree rooted at root.
        Returns: Node containing val if it exists, otherwise the Node for insertion.
        Time/Space: O(log n)/O(1)
        """

        # base case: node not found or the correct position for insertion found
        if root is None or root.value == val:
            return root

        # recursive search
        if val < root.value:
            if root.left is None:
                return root  # if the value is not present, return the current node as the insertion point
            return self.search(root.left, val)
        else:
            if root.right is None:
                return root  # if the value is not present, return the current node as the insertion point
            return self.search(root.right, val)

    def inorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Performs an inorder traversal (left, current, right) of the subtree rooted at root.
        Yields: Nodes of the subtree in inorder.
        Time/Space: O(n)/O(1)
        """

        if root is not None:
            yield from self.inorder(root.left)
            yield root
            yield from self.inorder(root.right)

    def __iter__(self) -> Generator[Node, None, None]:
        """
        Makes the AVL tree class iterable, enabling iteration over the tree in an inorder manner.
        Returns: A generator yielding the nodes of the tree in inorder.
        """

        yield from self.inorder(self.origin)

    def preorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Performs a preorder traversal (current, left, right) of the subtree rooted at root.
        Yields: Nodes of the subtree in preorder.
        Time/Space: O(n)/O(1)
        """

        if root is not None:
            yield root
            yield from self.preorder(root.left)
            yield from self.preorder(root.right)

    def postorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Performs a postorder traversal (left, right, current) of the subtree rooted at root.
        Yields: Nodes of the subtree in postorder.
        Time/Space: O(n)/O(1)
        """

        if root is not None:
            yield from self.postorder(root.left)
            yield from self.postorder(root.right)
            yield root

    def levelorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Performs a level-order (breadth-first) traversal of the subtree rooted at root.
        Yields: Nodes of the subtree in level-order.
        Time/Space: O(n)/O(n)
        """

        if root is None:
            return

        queue = SimpleQueue()
        queue.put(root)

        while not queue.empty():
            current = queue.get()
            yield current

            if current.left:
                queue.put(current.left)
            if current.right:
                queue.put(current.right)


####################################################################################################
# Given Code from Class #

class AVLWrappedDictionary:
    """
    Implementation of a helper class which will be used as tree node values in the
    NearestNeighborClassifier implementation.
    """
    # preallocate storage
    __slots__ = ["key", "dictionary"]

    def __init__(self, key: float) -> None:
        """
        Construct a AVLWrappedDictionary with a key to search/sort on and a dictionary to hold data.

        :param key: floating point key to be looked up by.
        """
        self.key = key
        self.dictionary = {}

    def __repr__(self) -> str:
        """
        Represent the AVLWrappedDictionary as a string.

        :return: string representation of the AVLWrappedDictionary.
        """
        pprinted_dict = json.dumps(self.dictionary, indent=2)
        return f"key: {self.key} dict:{self.dictionary}"

    def __str__(self) -> str:
        """
        Represent the AVLWrappedDictionary as a string.

        :return: string representation of the AVLWrappedDictionary.
        """
        return repr(self)

    def __eq__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement == operator to compare 2 AVLWrappedDictionaries by key only.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating whether keys of AVLWrappedDictionaries are equal
        """
        return abs(self.key - other.key) < 1e-6

    def __lt__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement < operator to compare 2 AVLWrappedDictionarys by key only.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating ordering of AVLWrappedDictionaries
        """
        return self.key < other.key and not abs(self.key - other.key) < 1e-6

    def __gt__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement > operator to compare 2 AVLWrappedDictionaries by key only.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating ordering of AVLWrappedDictionaries
        """
        return self.key > other.key and not abs(self.key - other.key) < 1e-6


class NearestNeighborClassifier:
    """
    Implementation of a one-dimensional nearest-neighbor classifier with AVL tree lookups.
    Modify only below indicated line.
    """
    # preallocate storage
    __slots__ = ["resolution", "tree"]

    def __init__(self, resolution: int) -> None:
        """
        Construct a one-dimensional nearest neighbor classifier with AVL tree lookups.
        Data are assumed to be floating point values in the closed interval [0, 1].

        :param resolution: number of decimal places the data will be rounded to, effectively
                           governing the capacity of the model - for example, with a resolution of
                           1, the classifier could maintain up to 11 nodes, spaced 0.1 apart - with
                           a resolution of 2, the classifier could maintain 101 nodes, spaced 0.01
                           apart, and so on - the maximum number of nodes is bounded by
                           10^(resolution) + 1.
        """
        self.tree = AVLTree()
        self.resolution = resolution

        # pre-construct lookup tree with AVLWrappedDictionary objects storing (key, dictionary)
        # pairs, but which compare with <, >, == on key only
        for i in range(10 ** resolution + 1):
            w_dict = AVLWrappedDictionary(key=(i / 10 ** resolution))
            self.tree.insert(self.tree.origin, w_dict)

    def __repr__(self) -> str:
        """
        Represent the NearestNeighborClassifier as a string.

        :return: string representation of the NearestNeighborClassifier.
        """
        return f"NNC(resolution={self.resolution}):\n{self.tree}"

    def __str__(self) -> str:
        """
        Represent the NearestNeighborClassifier as a string.

        :return: string representation of the NearestNeighborClassifier.
        """
        return repr(self)

    def visualize(self, filename: str = "nnc_visualization.svg") -> str:
        svg_string = svg(self.tree.origin, 48, nnc_mode=True)
        print(svg_string, file=open(filename, 'w'))
        return svg_string

 # Application Problem #

    def fit(self, data: List[Tuple[float, str]]) -> None:
        """
        Trains the classifier with a dataset to learn associations between features (x) and labels (y).
        Example with Seasons: (temperature, season)
        Updates the count of the label y in the corresponding dictionary.
        Time/Space: O(n log n) / O(n)
        Parameters: data (List[Tuple[float, str]]): List of (x, y) pairs.
        Returns: None
        """

        for x, y in data:
            # round the x value to the specified precision
            newx = round(x, self.resolution)

            # create an AVLWrappedDictionary object with the rounded x value
            wrap = AVLWrappedDictionary(newx)

            # find the corresponding node in the tree
            node = self.tree.search(self.tree.origin, wrap)

            if node is not None:
                # update the count of the label y in this dictionary
                node.value.dictionary[y] = node.value.dictionary.get(y, 0) + 1

    def predict(self, x: float, delta: float) -> str:
        """
        Predicts the label y for a given feature x within ± delta.
        Determines the most common label y across relevant dictionaries.
        Parameters: x (float): Feature value for prediction.
                    delta (float): Range for searching nodes in the tree.
        Returns: Optional[str]: Predicted label y, or None if uncertain.
        """

        # round the x value to the specified precision
        newx = round(x, self.resolution)

        labels = {}

        # calculate the number of steps based on the resolution and delta
        steps = int(delta * 10 ** self.resolution)

        # search for nodes in the tree whose keys are within ± delta of the rounded x value
        for i in range(-steps, steps + 1):
            # create an AVLWrappedDictionary object with the current x value
            wrap = AVLWrappedDictionary(newx + i / 10 ** self.resolution)

            node = self.tree.search(self.tree.origin, wrap)

            if node is not None:
                # update the label_counts with the labels in this node's dictionary
                for label, count in node.value.dictionary.items():
                    labels[label] = labels.get(label, 0) + count

        # if no labels found, return None
        if not labels:
            return None

        # return the most common label
        most_common_label = max(labels, key=labels.get)
        return most_common_label
