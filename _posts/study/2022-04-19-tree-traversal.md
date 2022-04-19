---
layout: post
title: "[알고리즘] 트리 순회(Tree Traversal)"
description: >
  트리 순회(Tree Traversal) 개념 정리

category:   study
tags:       algorithm
---

* this unordered seed list will be replaced by the toc
{:toc}

## 트리 순회(Tree Traversal)
트리 순회는 트리 구조에서 각각의 노드를 정확히 한 번만, 체계적인 방법으로 방문하는 과정을 말한다.[^1]
보통 이진 트리(Binary Tree)[^2]를 사용하기 때문에, 통상적으로 트리 순회는 이진 트리를 대상으로 한다.  

본 포스트의 코드에서, 트리의 노드는 아래와 같이 구현한다고 가정한다.

~~~py
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    def __repr__(self):
        return f"{self.val}"
~~~

트리 순회는 크게 *깊이 우선 탐색(DFS)*과 *너비 우선 탐색(BFS)* 두 종류로 나눌 수 있다.

## 깊이 우선 탐색(Depth First Search ,DFS)
*깊이 우선 탐색(DFS)*은 이름에서 알 수 있듯이 *깊이를 우선으로 순회*한다. 다시 말해, 루트 노드에서 시작하여 자식 노드들을 우선적으로 탐색하여 특정 리프에 도달한 뒤, 루트 노드로 돌아온 뒤 다른 브랜치를 탐색한다.  

깊이 우선 탐색은 더 나아가 루트노드, 왼쪽 자식 노드, 오른쪽 자식 노드의 방문 순서에 따라 **inorder**, **preorder**, **postorder**로 나뉜다.

* **Inorder**
    1. 왼쪽 서브트리를 순회한다. `Inorder(left-subtree)`
    2. 루트노드를 방문한다.
    3. 오른쪽 서브트리를 순회한다. `Inorder(right-subtree)`
* **Preorder**: Node -> Left -> Right
    1. 루트노드를 방문한다.
    2. 왼쪽 서브트리를 순회한다. `Preorder(left-subtree)`
    3. 오른쪽 서브트리를 순회한다. `Preorder(right-subtree)`
* **Postorder**: Left -> Right -> Node
    1. 왼쪽 서브트리를 순회한다. `Postorder(left-subtree)`
    2. 오른쪽 서브트리를 순회한다. `Postorder(right-subtree)`
    3. 루트노드를 방문한다. 

DFS는 재귀나 스택을 사용하여 구현할 수 있다. 

### 재귀를 사용한 구현
재귀를 사용한 구현은 간단하다. 각 정의에 따라 순서대로 함수를 재귀적으로 사용하면 된다.

~~~py
def inorder_recursive(node: TreeNode):
    if node is None:
        return
    inorder_recursive(node.left)
    print(node, end = " -> ")
    inorder_recursive(node.right)

def preorder_recursive(node: TreeNode):
    if node is None:
        return
    print(node, end = " -> ")
    preorder_recursive(node.left)
    preorder_recursive(node.right)

def postorder_recursive(node: TreeNode):
    if node is None:
        return
    postorder_recursive(node.left)
    postorder_recursive(node.right)
    print(node, end = " -> ")
~~~

### 스택을 사용한 구현

DFS 기반의 방법들은 스택을 사용하여 반복구조로 구현이 가능하다. 각각의 방법의 구현방법을 살펴보자.

**Inorder traversal**의 경우 스택을 사용하여 아래와 같은 알고리즘으로 구현할 수 있다.
1. 빈 스택 `S`를 생성한다.
2. 현재 방문 중인 노드 `node`를 트리의 root로 설정한다.
3. `node`가 `None`값이 될 때까지 `node = node.left` 해주면서 스택 `S`에 `node`를 넣어준다.
4. `node`가 `None`값이며 `S`가 비어있지 않으면
    1. 스택 `S`에서 item 하나를 pop한다.
    2. pop된 item을 print하고, `node = popped_item.right`로 설정한다.
    3. 3.으로 돌아간다.
5. `node`가 `None`이며 `S`가 비면 종료한다.

코드로 구현하면 아래와 같다.  

~~~py
# inorder traversalusing iteration
def inorder_iter(root: TreeNode):
    if root is None:
        return 

    stack, node = [], root
    while True:
        if node:
            stack.append(node)
            node = node.left
        elif stack:
            node = stack.pop()
            print(node, end=" -> ")
            node = node.right
        else:
            break

~~~

**Preorder traversal**의 경우도 스택을 사용해서 아래와 같은 알고리즘으로 구현할 수 있다.
1. 빈 스택 `S`를 생성하고 root노드를 스택에 넣어준다.
2. 스택 `S`가 빌 때까지 다음을 수행한다.
    1. 스택 `S`에서 item 하나를 pop하고 print 한다.
    2. pop된 item의 오른쪽 자식을 스택에 넣어준다.
    3. pop된 item의 왼쪽 자식을 스택에 넣어준다.

*오른쪽 자식을 왼쪽 자식보다 먼저 넣어주어* pop할때 왼쪽 서브트리 부터 순회하게 함에 주의하자. 코드는 아래와 같다.

~~~py
# preorder traversal using iteration
def preorder_iter(root: TreeNode):
    if root is None:
        return
    
    stack = [root]
    while stack:
        node = stack.pop()
        print(node, end=" -> ")
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
~~~

마지막으로 **Postorder traversal**의 경우 2개의 스택을 사용하여 구현할 수 있다. 하나의 스택을 사용하여 postorder traversal과 반대 방향으로 순회하여, 또다른 스택에 이 순회 과정을 넣어준다. 마지막으로 저장된 순서를 pop 하면서 출력해 주면 된다.

1. 빈 스택 `S1`과 `S2`를 생성하고, 스택 `S1`에 root를 넣어준다.
2. 스택 `S1`이 빌 때까지 다음을 수행한다.
    1. `S1`에서 하나의 item을 pop하고 이를 `S2`에 넣어준다.
    2. item의 왼쪽 자식과 오른쪽 자식을 `S1`에 넣어준다.
3. `S2`를 pop하면서 print한다.

코드는 아래와 같다.

~~~py
# postorder traversal using iteration
def postorder_iter(root: TreeNode):
    if root is None:
        return
    
    stack1, stack2 = [root], []
    while stack1:
        node = stack1.pop()
        stack2.append(node)
        if node.left:
            stack1.append(node.left)
        if node.right:
            stack1.append(node.right)
    while stack2:
        print(stack2.pop(), end=" -> ")

~~~

## 너비 우선 탐색 (Breadth First Search, BFS)
*너비 우선 탐색(BFS)*은 트리의 루트에서 리프까지 높이 순대로 (위에서 아래로) 순회한다. 이러한 성질 때문에 level-order traversal라고도 한다. 아래의 노드에 도달하기 전 반드시 위쪽의 노드들을 모두 순회해야 한다.

기본구조는 다음과 같다.
1. 루트노드를 방문한다.
2. 루트노드의 왼쪽 자식을 방문한다.
3. 루트노드의 오른쪽 자식을 방문한다.

BFS는 큐를 이용하여 구현한다.

~~~py
def bfs(root: TreeNode):
    if root is None:
        return 
    
    queue = [root]
    while queue:
        node = queue.pop(0)
        print(node, end = " -> ")
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
~~~

## 예시
![Binary Search Tree](\assets\img\blog\algorithm\bst_02.png)
{: .centered}

위와 같은 이진 트리가 있을 때, 각의 순회의 결과는 다음과 같다.
* Preorder: `16 -> 12 -> 11 -> 9 -> 8 -> 10 -> 13 -> 15 -> 20 -> 17 -> 23 ->`
* Inorder: `8 -> 9 -> 10 -> 11 -> 12 -> 13 -> 15 -> 16 -> 17 -> 20 -> 23 ->`
* Postorder: `8 -> 10 -> 9 -> 11 -> 15 -> 13 -> 12 -> 17 -> 23 -> 20 -> 16 ->`
* BFS: `16 -> 12 -> 20 -> 11 -> 13 -> 17 -> 23 -> 9 -> 15 -> 8 -> 10 ->`

## 참고글
* [https://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/](https://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/)
* [https://www.geeksforgeeks.org/inorder-tree-traversal-without-recursion/](https://www.geeksforgeeks.org/inorder-tree-traversal-without-recursion/)
* [https://www.geeksforgeeks.org/iterative-preorder-traversal/](https://www.geeksforgeeks.org/iterative-preorder-traversal/)
* [https://www.geeksforgeeks.org/iterative-postorder-traversal/](https://www.geeksforgeeks.org/iterative-postorder-traversal/)


[^1]: [위키피디아](https://ko.wikipedia.org/wiki/%ED%8A%B8%EB%A6%AC_%EC%88%9C%ED%9A%8C)에서의 트리 순회의 정의
[^2]: 이진 트리(Binary tree)란 자식 노드의 개수가 최대 2개인 트리를 말한다. 다른 트리에 비해 훨씬 간결할 뿐만 아니라 여러가지 알고리즘을 구현하는 일도 좀 더 간단하게 처리할 수 있어, 대체로 트리라고 하면 대부분 이진트리를 일컫는다.
