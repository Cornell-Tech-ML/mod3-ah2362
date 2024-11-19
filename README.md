# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


PARALLELIZATION CHECK:
(.venv) (base) arjunhegde@dhcp-vl2051-8 mod3-ah2362 % python project/parallel_check.py                        
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. 
Cornell Tech (MEng)/CS5781 - Machine Learning 
Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. Cornell Tech (MEng)/CS5781 - Machine Learning Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (163) 
-----------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                  | 
        out: Storage,                                                                          | 
        out_shape: Shape,                                                                      | 
        out_strides: Strides,                                                                  | 
        in_storage: Storage,                                                                   | 
        in_shape: Shape,                                                                       | 
        in_strides: Strides,                                                                   | 
    ) -> None:                                                                                 | 
        # Check if tensors are stride-aligned                                                  | 
        if np.array_equal(out_strides, in_strides) and np.array_equal(out_shape, in_shape):    | 
            for i in prange(len(out)):---------------------------------------------------------| #0
                out[i] = fn(in_storage[i])                                                     | 
            return                                                                             | 
        # If not stride-aligned, use indexing                                                  | 
        size = len(out)                                                                        | 
        for i in prange(size):-----------------------------------------------------------------| #1
            out_index = np.empty(len(out_shape), np.int32)                                     | 
            in_index = np.empty(len(in_shape), np.int32)                                       | 
            to_index(i, out_shape, out_index)                                                  | 
            broadcast_index(out_index, out_shape, in_shape, in_index)                          | 
            out[index_to_position(out_index, out_strides)] = fn(                               | 
                in_storage[index_to_position(in_index, in_strides)]                            | 
            )                                                                                  | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. 
Cornell Tech (MEng)/CS5781 - Machine Learning 
Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (179) is hoisted out of the 
parallel loop labelled #1 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. 
Cornell Tech (MEng)/CS5781 - Machine Learning 
Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (180) is hoisted out of the 
parallel loop labelled #1 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: in_index = np.empty(len(in_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. 
Cornell Tech (MEng)/CS5781 - Machine Learning 
Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (213)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. Cornell Tech (MEng)/CS5781 - Machine Learning Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (213) 
-----------------------------------------------------------------------|loop #ID
    def _zip(                                                          | 
        out: Storage,                                                  | 
        out_shape: Shape,                                              | 
        out_strides: Strides,                                          | 
        a_storage: Storage,                                            | 
        a_shape: Shape,                                                | 
        a_strides: Strides,                                            | 
        b_storage: Storage,                                            | 
        b_shape: Shape,                                                | 
        b_strides: Strides,                                            | 
    ) -> None:                                                         | 
        # Check if tensors are stride-aligned                          | 
        if (np.array_equal(out_strides, a_strides) and                 | 
            np.array_equal(out_strides, b_strides) and                 | 
            np.array_equal(out_shape, a_shape) and                     | 
            np.array_equal(out_shape, b_shape)):                       | 
            for i in prange(len(out)):---------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                | 
            return                                                     | 
                                                                       | 
        # If not stride-aligned, use broadcasting                      | 
        size = len(out)                                                | 
        for i in prange(size):-----------------------------------------| #3
            out_index = np.empty(len(out_shape), np.int32)             | 
            a_index = np.empty(len(a_shape), np.int32)                 | 
            b_index = np.empty(len(b_shape), np.int32)                 | 
            to_index(i, out_shape, out_index)                          | 
            broadcast_index(out_index, out_shape, a_shape, a_index)    | 
            broadcast_index(out_index, out_shape, b_shape, b_index)    | 
            out[index_to_position(out_index, out_strides)] = fn(       | 
                a_storage[index_to_position(a_index, a_strides)],      | 
                b_storage[index_to_position(b_index, b_strides)]       | 
            )                                                          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. 
Cornell Tech (MEng)/CS5781 - Machine Learning 
Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (236) is hoisted out of the 
parallel loop labelled #3 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. 
Cornell Tech (MEng)/CS5781 - Machine Learning 
Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (237) is hoisted out of the 
parallel loop labelled #3 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: a_index = np.empty(len(a_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. 
Cornell Tech (MEng)/CS5781 - Machine Learning 
Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (238) is hoisted out of the 
parallel loop labelled #3 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: b_index = np.empty(len(b_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. 
Cornell Tech (MEng)/CS5781 - Machine Learning 
Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (271)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. Cornell Tech (MEng)/CS5781 - Machine Learning Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (271) 
---------------------------------------------------------------------|loop #ID
    def _reduce(                                                     | 
        out: Storage,                                                | 
        out_shape: Shape,                                            | 
        out_strides: Strides,                                        | 
        a_storage: Storage,                                          | 
        a_shape: Shape,                                              | 
        a_strides: Strides,                                          | 
        reduce_dim: int,                                             | 
    ) -> None:                                                       | 
        # Calculate the size of the non-reduced dimensions           | 
        size = 1                                                     | 
        for i in range(len(out_shape)):                              | 
            size *= out_shape[i]                                     | 
                                                                     | 
        # Main parallel loop over non-reduced dimensions             | 
        for i in prange(size):---------------------------------------| #4
            # Convert position to indices for output                 | 
            out_index = np.empty(len(out_shape), np.int32)           | 
            to_index(i, out_shape, out_index)                        | 
                                                                     | 
            # Convert to position for output                         | 
            out_pos = index_to_position(out_index, out_strides)      | 
                                                                     | 
            # Inner loop over reduced dimension                      | 
            for j in range(a_shape[reduce_dim]):                     | 
                # Copy output index to get input index               | 
                a_index = np.empty(len(a_shape), np.int32)           | 
                for k in range(len(out_shape)):                      | 
                    a_index[k] = out_index[k]                        | 
                # Set the reduced dimension index                    | 
                a_index[reduce_dim] = j                              | 
                                                                     | 
                # Get input position and apply reduction             | 
                a_pos = index_to_position(a_index, a_strides)        | 
                out[out_pos] = fn(out[out_pos], a_storage[a_pos])    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. 
Cornell Tech (MEng)/CS5781 - Machine Learning 
Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (297) is hoisted out of the 
parallel loop labelled #4 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: a_index = np.empty(len(a_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. 
Cornell Tech (MEng)/CS5781 - Machine Learning 
Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (288) is hoisted out of the 
parallel loop labelled #4 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. 
Cornell Tech (MEng)/CS5781 - Machine Learning 
Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (310)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. Cornell Tech (MEng)/CS5781 - Machine Learning Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (310) 
-------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                       | 
    out: Storage,                                                  | 
    out_shape: Shape,                                              | 
    out_strides: Strides,                                          | 
    a_storage: Storage,                                            | 
    a_shape: Shape,                                                | 
    a_strides: Strides,                                            | 
    b_storage: Storage,                                            | 
    b_shape: Shape,                                                | 
    b_strides: Strides,                                            | 
) -> None:                                                         | 
    """NUMBA tensor matrix multiply function.                      | 
                                                                   | 
    Should work for any tensor shapes that broadcast as long as    | 
                                                                   | 
    ```                                                            | 
    assert a_shape[-1] == b_shape[-2]                              | 
    ```                                                            | 
                                                                   | 
    Optimizations:                                                 | 
                                                                   | 
    * Outer loop in parallel                                       | 
    * No index buffers or function calls                           | 
    * Inner loop should have no global writes, 1 multiply.         | 
                                                                   | 
                                                                   | 
    Args:                                                          | 
    ----                                                           | 
        out (Storage): storage for `out` tensor                    | 
        out_shape (Shape): shape for `out` tensor                  | 
        out_strides (Strides): strides for `out` tensor            | 
        a_storage (Storage): storage for `a` tensor                | 
        a_shape (Shape): shape for `a` tensor                      | 
        a_strides (Strides): strides for `a` tensor                | 
        b_storage (Storage): storage for `b` tensor                | 
        b_shape (Shape): shape for `b` tensor                      | 
        b_strides (Strides): strides for `b` tensor                | 
                                                                   | 
    Returns:                                                       | 
    -------                                                        | 
        None : Fills in `out`                                      | 
                                                                   | 
    """                                                            | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0         | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0         | 
                                                                   | 
    # Parallel over batch and i dimensions                         | 
    for batch in prange(out_shape[0]):-----------------------------| #5
        for i in range(out_shape[1]):                              | 
            for j in range(out_shape[2]):                          | 
                # Get output position                              | 
                out_pos = (                                        | 
                    batch * out_strides[0] +                       | 
                    i * out_strides[1] +                           | 
                    j * out_strides[2]                             | 
                )                                                  | 
                                                                   | 
                # Initialize accumulator                           | 
                acc = 0.0                                          | 
                                                                   | 
                # Inner loop - matrix multiply                     | 
                for k in range(a_shape[2]):                        | 
                    # Get positions in a and b                     | 
                    a_pos = (                                      | 
                        batch * a_batch_stride +                   | 
                        i * a_strides[1] +                         | 
                        k * a_strides[2]                           | 
                    )                                              | 
                    b_pos = (                                      | 
                        batch * b_batch_stride +                   | 
                        k * b_strides[1] +                         | 
                        j * b_strides[2]                           | 
                    )                                              | 
                                                                   | 
                    # Multiply and accumulate                      | 
                    acc += a_storage[a_pos] * b_storage[b_pos]     | 
                                                                   | 
                # Store result                                     | 
                out[out_pos] = acc                                 | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None



GPU SPLIT: 

Epoch  0  loss  9.730172629290472 correct 21 avg_time 3.5570s
Epoch  10  loss  3.365466269405985 correct 38 avg_time 1.2210s
Epoch  20  loss  6.045775214563723 correct 42 avg_time 1.2110s
Epoch  30  loss  3.7537039879188905 correct 44 avg_time 1.2047s
Epoch  40  loss  3.222153672416769 correct 48 avg_time 1.1945s
Epoch  50  loss  3.3337060243052687 correct 50 avg_time 1.2002s
Epoch  60  loss  2.638331331889649 correct 49 avg_time 1.1965s
Epoch  70  loss  1.2471833665580965 correct 50 avg_time 1.1938s
Epoch  80  loss  0.8578205222503378 correct 50 avg_time 1.1986s
Epoch  90  loss  1.0600947054068055 correct 50 avg_time 1.2139s
Epoch  100  loss  1.0439841527964597 correct 50 avg_time 1.2075s
Epoch  110  loss  0.9957452341717226 correct 50 avg_time 1.2062s
Epoch  120  loss  0.5092253524948086 correct 50 avg_time 1.2070s
Epoch  130  loss  0.8829190369669233 correct 50 avg_time 1.2099s
Epoch  140  loss  0.2545767841228047 correct 50 avg_time 1.2092s
Epoch  150  loss  0.5937601724255802 correct 50 avg_time 1.2042s
Epoch  160  loss  1.1986015354576112 correct 50 avg_time 1.2006s
Epoch  170  loss  0.819681791520827 correct 50 avg_time 1.1938s
Epoch  180  loss  0.7330711538508179 correct 50 avg_time 1.1927s
Epoch  190  loss  0.6057783414283815 correct 50 avg_time 1.1962s
Epoch  200  loss  0.6166619638435488 correct 50 avg_time 1.1874s
Epoch  210  loss  0.32678503554725213 correct 50 avg_time 1.1879s
Epoch  220  loss  0.5124734791427106 correct 50 avg_time 1.1957s
Epoch  230  loss  0.38848004156046295 correct 50 avg_time 1.1952s
Epoch  240  loss  0.21786825345767008 correct 50 avg_time 1.2048s
Epoch  250  loss  0.7971541224914944 correct 50 avg_time 1.2054s
Epoch  260  loss  0.04914889533882441 correct 50 avg_time 1.2063s
Epoch  270  loss  0.288516640433794 correct 50 avg_time 1.1894s
Epoch  280  loss  0.3443898740598669 correct 50 avg_time 1.1989s
Epoch  290  loss  0.4510832390420654 correct 50 avg_time 1.1962s
Epoch  300  loss  0.2624149353119811 correct 50 avg_time 1.1908s
Epoch  310  loss  0.3117191334140345 correct 50 avg_time 1.1939s
Epoch  320  loss  0.18565351704632166 correct 50 avg_time 1.1918s
Epoch  330  loss  0.4548046278716599 correct 50 avg_time 1.1919s
Epoch  340  loss  0.3181655405055055 correct 50 avg_time 1.1977s
Epoch  350  loss  0.02080773324656941 correct 50 avg_time 1.2036s
Epoch  360  loss  0.07223048524870604 correct 50 avg_time 1.1978s
Epoch  370  loss  0.32323477539694007 correct 50 avg_time 1.1996s
Epoch  380  loss  0.03323711905284609 correct 50 avg_time 1.1904s
Epoch  390  loss  0.2206777294129984 correct 50 avg_time 1.1988s
Epoch  400  loss  0.08455042975435852 correct 50 avg_time 1.1806s
Epoch  410  loss  0.10236154323909237 correct 50 avg_time 1.1838s
Epoch  420  loss  0.055178456189982246 correct 50 avg_time 1.1889s
Epoch  430  loss  0.04106630917808682 correct 50 avg_time 1.1752s
Epoch  440  loss  0.17893066220593076 correct 50 avg_time 1.1817s
Epoch  450  loss  0.03908642040087393 correct 50 avg_time 1.1875s
Epoch  460  loss  0.08453786375321957 correct 50 avg_time 1.1833s
Epoch  470  loss  0.3625463507632614 correct 50 avg_time 1.2000s
Epoch  480  loss  0.07015643105303025 correct 50 avg_time 1.1955s
Epoch  490  loss  0.31045473118691624 correct 50 avg_time 1.1946s

GPU XOR:

Epoch  0  loss  7.214904900073069 correct 30 avg_time 4.2753s
Epoch  10  loss  5.9539565697949275 correct 36 avg_time 1.2085s
Epoch  20  loss  3.8554198058473412 correct 45 avg_time 1.2203s
Epoch  30  loss  3.054050936226489 correct 44 avg_time 1.2036s
Epoch  40  loss  3.9502316053639985 correct 44 avg_time 1.1914s
Epoch  50  loss  2.4622526329789203 correct 45 avg_time 1.2034s
Epoch  60  loss  3.386435526741054 correct 45 avg_time 1.1980s
Epoch  70  loss  3.3356534583584327 correct 45 avg_time 1.1874s
Epoch  80  loss  2.0296239917412318 correct 46 avg_time 1.1979s
Epoch  90  loss  3.307179349668927 correct 46 avg_time 1.1991s
Epoch  100  loss  2.9781864593415373 correct 46 avg_time 1.2029s
Epoch  110  loss  1.5491403841041274 correct 47 avg_time 1.1954s
Epoch  120  loss  3.078270739235293 correct 47 avg_time 1.1997s
Epoch  130  loss  1.3678068056387116 correct 47 avg_time 1.2078s
Epoch  140  loss  1.4491132030075937 correct 47 avg_time 1.1873s
Epoch  150  loss  1.083130788026784 correct 48 avg_time 1.1971s
Epoch  160  loss  0.7204440289962457 correct 46 avg_time 1.1946s
Epoch  170  loss  2.018061908552479 correct 49 avg_time 1.1725s
Epoch  180  loss  1.659044541596597 correct 49 avg_time 1.1969s
Epoch  190  loss  2.133072845080946 correct 47 avg_time 1.1910s
Epoch  200  loss  2.1273348318422105 correct 48 avg_time 1.1854s
Epoch  210  loss  0.4118087588296476 correct 47 avg_time 1.1894s
Epoch  220  loss  0.3909522496735416 correct 47 avg_time 1.1811s
Epoch  230  loss  0.5696522835913491 correct 48 avg_time 1.1959s
Epoch  240  loss  2.228621759863754 correct 49 avg_time 1.1784s
Epoch  250  loss  0.22318813591352288 correct 48 avg_time 1.2090s
Epoch  260  loss  1.4187766254197745 correct 48 avg_time 1.1930s
Epoch  270  loss  0.8574860955803534 correct 47 avg_time 1.2037s
Epoch  280  loss  0.8313921314124499 correct 48 avg_time 1.2089s
Epoch  290  loss  0.15427382664764844 correct 48 avg_time 1.2359s
Epoch  300  loss  1.1926515460065794 correct 49 avg_time 1.2120s
Epoch  310  loss  0.2264838962577828 correct 48 avg_time 1.2060s
Epoch  320  loss  0.5874236899648503 correct 50 avg_time 1.2046s
Epoch  330  loss  0.042582837451297505 correct 49 avg_time 1.2042s
Epoch  340  loss  0.10131388746241635 correct 49 avg_time 1.2019s
Epoch  350  loss  1.6845966046747216 correct 48 avg_time 1.2018s
Epoch  360  loss  2.253053447638478 correct 49 avg_time 1.1977s
Epoch  370  loss  1.340037401182823 correct 49 avg_time 1.1940s
Epoch  380  loss  1.1145352025056772 correct 48 avg_time 1.1995s
Epoch  390  loss  1.4396773093124067 correct 49 avg_time 1.2081s
Epoch  400  loss  1.4906739334916943 correct 50 avg_time 1.2022s
Epoch  410  loss  0.8562324961505383 correct 49 avg_time 1.1977s
Epoch  420  loss  0.23833385073683008 correct 49 avg_time 1.1871s
Epoch  430  loss  0.31180371574794985 correct 49 avg_time 1.1928s
Epoch  440  loss  0.21390801187742717 correct 50 avg_time 1.1901s
Epoch  450  loss  0.4331039127247256 correct 49 avg_time 1.1919s
Epoch  460  loss  1.4654048489672447 correct 48 avg_time 1.1822s
Epoch  470  loss  0.44929540384311983 correct 49 avg_time 1.1852s
Epoch  480  loss  0.8046085695214704 correct 49 avg_time 1.1956s
Epoch  490  loss  0.24591628115004016 correct 50 avg_time 1.1914s

GPU SPLIT:


CPU SPLIT:
Epoch  0  loss  7.593465113149294 correct 29 avg_time 14.8155s
Epoch  10  loss  6.195920404153672 correct 37 avg_time 0.0881s
Epoch  20  loss  4.4147028550024965 correct 41 avg_time 0.0871s
Epoch  30  loss  5.0226974033745435 correct 44 avg_time 0.0888s
Epoch  40  loss  6.160467548849999 correct 45 avg_time 0.0874s
Epoch  50  loss  4.6281274991024945 correct 47 avg_time 0.0877s
Epoch  60  loss  3.9226165628156497 correct 43 avg_time 0.0980s
Epoch  70  loss  2.882376955473437 correct 46 avg_time 0.1536s
Epoch  80  loss  2.1577527863626136 correct 47 avg_time 0.1232s
Epoch  90  loss  1.8885497392743695 correct 48 avg_time 0.0885s
Epoch  100  loss  2.2604598829232176 correct 48 avg_time 0.0867s
Epoch  110  loss  1.7561184903229294 correct 48 avg_time 0.0866s
Epoch  120  loss  1.2373424943620706 correct 47 avg_time 0.0896s
Epoch  130  loss  1.8854033246027762 correct 50 avg_time 0.0894s
Epoch  140  loss  2.1714935298086813 correct 50 avg_time 0.0921s
Epoch  150  loss  1.6249677060854397 correct 47 avg_time 0.0879s
Epoch  160  loss  0.8790286385212366 correct 48 avg_time 0.0876s
Epoch  170  loss  0.9088959639708893 correct 50 avg_time 0.0875s
Epoch  180  loss  1.272563527124825 correct 50 avg_time 0.0875s
Epoch  190  loss  1.598646238391296 correct 48 avg_time 0.1089s
Epoch  200  loss  1.0626426233198047 correct 50 avg_time 0.1812s
Epoch  210  loss  1.1014365924833074 correct 50 avg_time 0.0870s
Epoch  220  loss  1.3777656472069435 correct 50 avg_time 0.0876s
Epoch  230  loss  0.8367502008456416 correct 50 avg_time 0.0883s
Epoch  240  loss  0.5475628130013186 correct 50 avg_time 0.0887s
Epoch  250  loss  0.21297960118903944 correct 50 avg_time 0.0875s
Epoch  260  loss  0.7920743617769147 correct 50 avg_time 0.0872s
Epoch  270  loss  1.10573162873563 correct 50 avg_time 0.0859s
Epoch  280  loss  0.47464892954354504 correct 50 avg_time 0.0858s
Epoch  290  loss  0.29235167844818066 correct 50 avg_time 0.0887s
Epoch  300  loss  1.1329733185120794 correct 50 avg_time 0.0863s
Epoch  310  loss  1.1430811466866904 correct 50 avg_time 0.0863s
Epoch  320  loss  0.8029735167520787 correct 50 avg_time 0.1310s
Epoch  330  loss  1.0491016755773417 correct 50 avg_time 0.1477s
Epoch  340  loss  0.16009307379564708 correct 50 avg_time 0.0856s
Epoch  350  loss  0.6414023948071151 correct 50 avg_time 0.0873s
Epoch  360  loss  0.6445569425428145 correct 50 avg_time 0.0887s
Epoch  370  loss  0.6447741549577456 correct 50 avg_time 0.0869s
Epoch  380  loss  0.5847339723949759 correct 50 avg_time 0.0881s
Epoch  390  loss  0.6700164161948058 correct 50 avg_time 0.0873s
Epoch  400  loss  1.0866146156735317 correct 49 avg_time 0.0875s
Epoch  410  loss  0.6530979069539821 correct 50 avg_time 0.0863s
Epoch  420  loss  0.5474481642237082 correct 50 avg_time 0.0869s
Epoch  430  loss  0.655244673671132 correct 50 avg_time 0.0869s
Epoch  440  loss  0.40264662497596787 correct 50 avg_time 0.0863s
Epoch  450  loss  0.0656247630914448 correct 50 avg_time 0.1312s
Epoch  460  loss  0.22042525253587664 correct 50 avg_time 0.1428s
Epoch  470  loss  0.19555266233193835 correct 50 avg_time 0.0870s
Epoch  480  loss  0.1362713872732746 correct 50 avg_time 0.0889s
Epoch  490  loss  0.6735020976226861 correct 50 avg_time 0.0877s

CPU XOR:
Epoch  0  loss  7.1289160736365735 correct 36 avg_time 14.9377s
Epoch  10  loss  3.9659662643202616 correct 45 avg_time 0.0873s
Epoch  20  loss  4.081161377674962 correct 45 avg_time 0.0866s
Epoch  30  loss  3.533309254468832 correct 42 avg_time 0.0860s
Epoch  40  loss  1.5577180812105011 correct 45 avg_time 0.0852s
Epoch  50  loss  1.596217635381318 correct 45 avg_time 0.0881s
Epoch  60  loss  1.511528735962185 correct 46 avg_time 0.0950s
Epoch  70  loss  1.6604456270892036 correct 46 avg_time 0.0857s
Epoch  80  loss  5.045064380136699 correct 47 avg_time 0.0948s
Epoch  90  loss  1.7796910766556326 correct 47 avg_time 0.1512s
Epoch  100  loss  1.2394942513340717 correct 48 avg_time 0.1230s
Epoch  110  loss  0.8635679013593611 correct 46 avg_time 0.0869s
Epoch  120  loss  1.1242726054705956 correct 48 avg_time 0.0864s
Epoch  130  loss  2.0735683501746123 correct 48 avg_time 0.0864s
Epoch  140  loss  1.5922509033041434 correct 48 avg_time 0.0869s
Epoch  150  loss  1.9822901978936207 correct 47 avg_time 0.0873s
Epoch  160  loss  0.986028560957503 correct 48 avg_time 0.0872s
Epoch  170  loss  0.8407158315379764 correct 48 avg_time 0.0876s
Epoch  180  loss  3.0571967453505744 correct 48 avg_time 0.0847s
Epoch  190  loss  3.1250924847947457 correct 48 avg_time 0.0858s
Epoch  200  loss  0.9845935894295175 correct 48 avg_time 0.0905s
Epoch  210  loss  2.3363609046935068 correct 48 avg_time 0.1041s
Epoch  220  loss  1.187786013318619 correct 48 avg_time 0.1605s
Epoch  230  loss  1.3802089050601336 correct 49 avg_time 0.1039s
Epoch  240  loss  0.4162744273004798 correct 49 avg_time 0.0843s
Epoch  250  loss  0.2565899877128124 correct 47 avg_time 0.0836s
Epoch  260  loss  2.181663424440431 correct 49 avg_time 0.0839s
Epoch  270  loss  0.21729465468576098 correct 48 avg_time 0.0880s
Epoch  280  loss  1.674478564747109 correct 48 avg_time 0.0865s
Epoch  290  loss  1.4543289069309129 correct 49 avg_time 0.0862s
Epoch  300  loss  0.8532959504373026 correct 48 avg_time 0.0857s
Epoch  310  loss  1.399386615311467 correct 48 avg_time 0.0846s
Epoch  320  loss  1.0670019166418938 correct 49 avg_time 0.0893s
Epoch  330  loss  0.45141760958125965 correct 49 avg_time 0.0868s
Epoch  340  loss  1.2918276346186783 correct 48 avg_time 0.0994s
Epoch  350  loss  0.1326022372332314 correct 49 avg_time 0.1617s
Epoch  360  loss  0.9819905644080886 correct 48 avg_time 0.1043s
Epoch  370  loss  0.11320866319779016 correct 49 avg_time 0.0867s
Epoch  380  loss  0.5266111475458133 correct 49 avg_time 0.0852s
Epoch  390  loss  1.434362327021645 correct 49 avg_time 0.0862s
Epoch  400  loss  0.9506406197602483 correct 48 avg_time 0.0863s
Epoch  410  loss  1.943539467611942 correct 48 avg_time 0.0856s
Epoch  420  loss  1.33123659891303 correct 49 avg_time 0.0864s
Epoch  430  loss  0.40233941487270064 correct 49 avg_time 0.0872s
Epoch  440  loss  0.30563599861953505 correct 49 avg_time 0.0878s
Epoch  450  loss  0.7211583809227131 correct 49 avg_time 0.0854s
Epoch  460  loss  1.5582165286235814 correct 50 avg_time 0.0857s
Epoch  470  loss  1.2109993812378583 correct 49 avg_time 0.1051s
Epoch  480  loss  0.8306202034234899 correct 49 avg_time 0.1664s
Epoch  490  loss  0.2797296629950458 correct 49 avg_time 0.0907s