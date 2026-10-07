# Data Structures — Lecture Notes

## Binary Search Trees
A binary search tree (BST) stores keys so that for every node, all keys in the left
subtree are smaller and all keys in the right subtree are larger. Search, insert and
delete take O(h) time where h is the height. A balanced BST has h = O(log n), but a
BST built from sorted input degenerates into a linked list with h = O(n).

In-order traversal of a BST visits keys in sorted order.

---

## B-Trees
A B-tree of minimum degree t is a balanced search tree designed for disks and other
block storage. Every node except the root holds between t-1 and 2t-1 keys, and every
internal node with k keys has k+1 children. All leaves are at the same depth.

Because each node holds many keys, the height is O(log_t n), so a lookup touches very
few disk blocks. Databases and file systems (e.g. PostgreSQL indexes, NTFS) use B-trees
or B+ trees. In a B+ tree, all values live in the leaves and leaves are linked for fast
range scans.

Insertion splits a full node (2t-1 keys) around its median key, pushing the median up
into the parent. If the root splits, the tree grows in height by one.

---

## Hash Tables
A hash table maps keys to buckets with a hash function. With a good hash function and
load factor alpha = n/m kept bounded, search, insert and delete take O(1) expected time.
Collisions are resolved by chaining (linked lists per bucket) or open addressing
(linear probing, quadratic probing, double hashing). When alpha grows past a threshold
(often 0.75) the table is resized, usually doubling m, which costs O(n) but is O(1)
amortized per insertion.
