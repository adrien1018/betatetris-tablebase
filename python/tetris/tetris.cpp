#include "tetris.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL TETRIS_PY_ARRAY_SYMBOL_
#include <numpy/ndarrayobject.h>

#include "board.h"

namespace {

int ParsePieceID(PyObject* obj) {
  if (PyUnicode_Check(obj)) {
    if (PyUnicode_GET_LENGTH(obj) < 1) {
      PyErr_SetString(PyExc_KeyError, "Invalid piece symbol.");
      return -1;
    }
    switch (PyUnicode_READ_CHAR(obj, 0)) {
      case 'T': return 0;
      case 'J': return 1;
      case 'Z': return 2;
      case 'O': return 3;
      case 'S': return 4;
      case 'L': return 5;
      case 'I': return 6;
      default: {
        PyErr_SetString(PyExc_KeyError, "Invalid piece symbol.");
        return -1;
      }
    }
  } else if (PyLong_Check(obj)) {
    long x = PyLong_AsLong(obj);
    if (x < 0 || x >= 7) {
      PyErr_SetString(PyExc_IndexError, "Piece ID out of range.");
      return -1;
    }
    return x;
  } else {
    PyErr_SetString(PyExc_TypeError, "Invalid type for piece.");
    return -1;
  }
}

bool CheckBoard(Board& board, PyObject* obj) {
  if (obj) {
    if (!PyObject_IsInstance(obj, (PyObject*)&py_board_class)) {
      PyErr_SetString(PyExc_TypeError, "Invalid board type.");
      return false;
    }
    board = reinterpret_cast<PythonBoard*>(obj)->board;
  }
  return true;
}

void TetrisDealloc(PythonTetris* self) {
  self->~PythonTetris();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject* TetrisNew(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  PythonTetris* self = (PythonTetris*)type->tp_alloc(type, 0);
  // leave initialization to __init__
  return (PyObject*)self;
}

int TetrisInit(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char *kwlist[] = {"seed", nullptr};
  unsigned long long seed = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|K", (char**)kwlist, &seed)) {
    return -1;
  }
  new(self) PythonTetris(seed);
  return 0;
}

PyObject* Tetris_InputPlacement(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"rotate", "x", "y", nullptr};
  int rotate, x, y;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iii", (char**)kwlist, &rotate, &x, &y)) {
    return nullptr;
  }
  auto [reward, raw_reward] = self->InputPlacement({rotate, x, y});
  PyObject* r1 = PyFloat_FromDouble(reward);
  PyObject* r2 = PyFloat_FromDouble(raw_reward);
  PyObject* ret = PyTuple_Pack(2, r1, r2);
  Py_DECREF(r1);
  Py_DECREF(r2);
  return ret;
}

PyObject* Tetris_Reset(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {
    "now_piece", "next_piece", "lines", "board",
    nullptr
  };
  PyObject *now_obj, *next_obj;
  int lines = 0;
  Board board = Board::Ones;
  PyObject* board_obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|i", (char**)kwlist,
        &now_obj, &next_obj, &lines, &board_obj)) {
    return nullptr;
  }
  if (!CheckBoard(board, board_obj)) return nullptr;
  int now_piece = ParsePieceID(now_obj);
  if (now_piece < 0) return nullptr;
  int next_piece = ParsePieceID(next_obj);
  if (next_piece < 0) return nullptr;
  self->Reset(board, lines, now_piece, next_piece);
  Py_RETURN_NONE;
}

PyObject* Tetris_ResetRandom(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {
    "lines", "board",
    nullptr
  };
  int lines = -1;
  Board board = Board::Ones;
  PyObject* board_obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iO", (char**)kwlist, &lines, &board_obj)) {
    return nullptr;
  }
  if (!CheckBoard(board, board_obj)) return nullptr;
  if (lines == -1) {
    self->Reset(board);
  } else {
    self->Reset(board, lines);
  }
  Py_RETURN_NONE;
}

PyObject* Tetris_GetState(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  PythonTetris::State state;
  self->GetState(state);
  PyObject *r1, *r2, *r3, *r4, *r5;
  {
    npy_intp dims[] = {state.board.size(), state.board[0].size(), state.board[0][0].size()};
    r1 = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    memcpy(PyArray_DATA((PyArrayObject*)r1), state.board.data(), sizeof(state.board));
  }
  {
    npy_intp dims[] = {state.meta.size()};
    r2 = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    memcpy(PyArray_DATA((PyArrayObject*)r2), state.meta.data(), sizeof(state.meta));
  }
  {
    npy_intp dims[] = {state.moves.size(), state.moves[0].size(), state.moves[0][0].size()};
    r3 = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    memcpy(PyArray_DATA((PyArrayObject*)r3), state.moves.data(), sizeof(state.moves));
  }
  {
    npy_intp dims[] = {state.move_meta.size()};
    r4 = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    memcpy(PyArray_DATA((PyArrayObject*)r4), state.move_meta.data(), sizeof(state.move_meta));
  }
  {
    npy_intp dims[] = {state.meta_int.size()};
    r5 = PyArray_SimpleNew(1, dims, NPY_INT32);
    memcpy(PyArray_DATA((PyArrayObject*)r5), state.meta_int.data(), sizeof(state.meta_int));
  }
  PyObject* ret = PyTuple_Pack(5, r1, r2, r3, r4, r5);
  Py_DECREF(r1);
  Py_DECREF(r2);
  Py_DECREF(r3);
  Py_DECREF(r4);
  Py_DECREF(r5);
  return ret;
}

static PyObject* Tetris_StateShapes(void*, PyObject* Py_UNUSED(ignored)) {
  PyObject *r1, *r2, *r3, *r4, *r5;
  {
    int dim1 = std::tuple_size<decltype(PythonTetris::State::board)>::value;
    int dim2 = std::tuple_size<decltype(PythonTetris::State::board)::value_type>::value;
    int dim3 = std::tuple_size<decltype(PythonTetris::State::board)::value_type::value_type>::value;
    r1 = Py_BuildValue("(iii)", dim1, dim2, dim3);
  }
  {
    int dim1 = std::tuple_size<decltype(PythonTetris::State::meta)>::value;
    r2 = Py_BuildValue("(i)", dim1);
  }
  {
    int dim1 = std::tuple_size<decltype(PythonTetris::State::moves)>::value;
    int dim2 = std::tuple_size<decltype(PythonTetris::State::moves)::value_type>::value;
    int dim3 = std::tuple_size<decltype(PythonTetris::State::moves)::value_type::value_type>::value;
    r3 = Py_BuildValue("(iii)", dim1, dim2, dim3);
  }
  {
    int dim1 = std::tuple_size<decltype(PythonTetris::State::move_meta)>::value;
    r4 = Py_BuildValue("(i)", dim1);
  }
  {
    int dim1 = std::tuple_size<decltype(PythonTetris::State::meta_int)>::value;
    r5 = Py_BuildValue("(i)", dim1);
  }
  PyObject* ret = PyTuple_Pack(5, r1, r2, r3, r4, r5);
  Py_DECREF(r1);
  Py_DECREF(r2);
  Py_DECREF(r3);
  Py_DECREF(r4);
  Py_DECREF(r5);
  return ret;
}

static PyObject* Tetris_StateTypes(void*, PyObject* Py_UNUSED(ignored)) {
  return Py_BuildValue("(sssss)", "float32", "float32", "float32", "float32", "int32");
}

PyObject* Tetris_IsOver(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyBool_FromLong((long)self->tetris.IsOver());
}

PyObject* Tetris_GetBoard(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  PythonBoard* board = PyObject_New(PythonBoard, &py_board_class);
  new(board) PythonBoard(self->tetris.GetBoard());
  return reinterpret_cast<PyObject*>(board);
}

PyObject* Tetris_GetRunScore(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->tetris.RunScore());
}

PyObject* Tetris_GetRunLines(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->tetris.RunLines());
}

PyObject* Tetris_GetRunPieces(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->tetris.RunPieces());
}

PyMethodDef py_tetris_class_methods[] = {
    {"IsOver", (PyCFunction)Tetris_IsOver, METH_NOARGS,
     "Check whether the game is over"},
    {"InputPlacement", (PyCFunction)Tetris_InputPlacement,
     METH_VARARGS | METH_KEYWORDS, "Input a placement and return the reward"},
    {"GetState", (PyCFunction)Tetris_GetState, METH_NOARGS, "Get state tuple"},
    {"StateShapes", (PyCFunction)Tetris_StateShapes, METH_NOARGS | METH_STATIC,
     "Get shapes of state array (static)"},
    {"StateTypes", (PyCFunction)Tetris_StateTypes, METH_NOARGS | METH_STATIC,
     "Get types of state array (static)"},
    {"Reset", (PyCFunction)Tetris_Reset, METH_VARARGS | METH_KEYWORDS,
     "Reset game and assign pieces randomly"},
    {"ResetRandom", (PyCFunction)Tetris_ResetRandom, METH_VARARGS | METH_KEYWORDS,
     "Reset game and assign pieces randomly"},
    {"GetBoard", (PyCFunction)Tetris_GetBoard, METH_NOARGS, "Get board object"},
    {"GetRunScore", (PyCFunction)Tetris_GetRunScore, METH_NOARGS, "Get score of this run"},
    {"GetRunLines", (PyCFunction)Tetris_GetRunLines, METH_NOARGS, "Get lines of this run"},
    {"GetRunPieces", (PyCFunction)Tetris_GetRunPieces, METH_NOARGS, "Get pieces of this run"},
    {nullptr}};

} // namespace

PyTypeObject py_tetris_class = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "tetris.Tetris",         // tp_name
    sizeof(PythonTetris),    // tp_basicsize
    0,                       // tp_itemsize
    (destructor)TetrisDealloc, // tp_dealloc
    0,                       // tp_print
    0,                       // tp_getattr
    0,                       // tp_setattr
    0,                       // tp_reserved
    0,                       // tp_repr
    0,                       // tp_as_number
    0,                       // tp_as_sequence
    0,                       // tp_as_mapping
    0,                       // tp_hash
    0,                       // tp_call
    0,                       // tp_str
    0,                       // tp_getattro
    0,                       // tp_setattro
    0,                       // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // tp_flags
    "Tetris class",          // tp_doc
    0,                       // tp_traverse
    0,                       // tp_clear
    0,                       // tp_richcompare
    0,                       // tp_weaklistoffset
    0,                       // tp_iter
    0,                       // tp_iternext
    py_tetris_class_methods, // tp_methods
    0,                       // tp_members
    0,                       // tp_getset
    0,                       // tp_base
    0,                       // tp_dict
    0,                       // tp_descr_get
    0,                       // tp_descr_set
    0,                       // tp_dictoffset
    (initproc)TetrisInit,    // tp_init
    0,                       // tp_alloc
    TetrisNew,               // tp_new
};
