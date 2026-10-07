# CS201 Midterm — Answer Key

Q1 answer key (10 marks): The four Coffman conditions — mutual exclusion (2), hold and
wait (2), no preemption (2), circular wait (2). Prevention (2 marks): break any one
condition, e.g. impose a global lock ordering to prevent circular wait, or require
processes to request all resources at once to prevent hold and wait.

Q2 answer key (10 marks): FCFS waiting times P1 = 0, P2 = 24, P3 = 27, average = 51/3 = 17.
SJF order P2, P3, P1: waiting times P2 = 0, P3 = 3, P1 = 6, average = 9/3 = 3.
5 marks each; deduct 2 for an arithmetic slip with correct method.

---

Q3 answer key (10 marks): Worst-case BST height is O(n) (n - 1 edges), occurring when
keys are inserted in sorted (or reverse-sorted) order so the tree degenerates into a
linked list (5). A B-tree with minimum degree t has height O(log_t n), more precisely
h <= log_t((n + 1) / 2) (5).
