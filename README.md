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


DEMONSTRATION OF PARALLEL CHECK OUTPUT:


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
                                                                                               | 
        # If not stride-aligned, use indexing                                                  | 
        size = len(out)                                                                        | 
        # Hoist allocations out of parallel loop                                               | 
        out_index = np.empty(len(out_shape), np.int32)                                         | 
        in_index = np.empty(len(in_shape), np.int32)                                           | 
                                                                                               | 
        for i in prange(size):-----------------------------------------------------------------| #1
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
No allocation hoisting found
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. 
Cornell Tech (MEng)/CS5781 - Machine Learning 
Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (216)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. Cornell Tech (MEng)/CS5781 - Machine Learning Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (216) 
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
        # Hoist allocations out of parallel loop                       | 
        out_index = np.empty(len(out_shape), np.int32)                 | 
        a_index = np.empty(len(a_shape), np.int32)                     | 
        b_index = np.empty(len(b_shape), np.int32)                     | 
                                                                       | 
        for i in prange(size):-----------------------------------------| #3
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
No allocation hoisting found
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. 
Cornell Tech (MEng)/CS5781 - Machine Learning 
Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (276)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. Cornell Tech (MEng)/CS5781 - Machine Learning Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (276) 
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
        # Hoist allocations out of parallel loop                     | 
        out_index = np.empty(len(out_shape), np.int32)               | 
        a_index = np.empty(len(a_shape), np.int32)                   | 
                                                                     | 
        # Main parallel loop over non-reduced dimensions             | 
        for i in prange(size):---------------------------------------| #4
            # Convert position to indices for output                 | 
            to_index(i, out_shape, out_index)                        | 
                                                                     | 
            # Convert to position for output                         | 
            out_pos = index_to_position(out_index, out_strides)      | 
                                                                     | 
            # Inner loop over reduced dimension                      | 
            for j in range(a_shape[reduce_dim]):                     | 
                # Copy output index to get input index               | 
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
No allocation hoisting found
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. 
Cornell Tech (MEng)/CS5781 - Machine Learning 
Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (317)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/arjunhegde/Library/CloudStorage/OneDrive-CornellUniversity/College/5. Cornell Tech (MEng)/CS5781 - Machine Learning Engineering/Repos/mod3-ah2362/minitorch/fast_ops.py (317) 
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
    # Parallel over batch dimension only                           | 
    for batch in prange(out_shape[0]):-----------------------------| #5
        # Serial over i dimension as per optimization output       | 
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