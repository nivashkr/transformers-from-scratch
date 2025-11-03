#include "utils/Tensor.h"
#include "utils/Helpers.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <memory>

void test_addition()
{
    std::cout << "\nTest Case: Addition" << std::endl;
    auto scalar_like = Tensor::create(std::vector<int>{1, 1}, std::make_shared<std::vector<float>>(std::vector<float>{2.0}));
    auto matrix_2d = Tensor::create(std::vector<int>{2, 3}, std::make_shared<std::vector<float>>(std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
    auto add_result1 = (*scalar_like) + matrix_2d;
    assert(add_result1->get_shape() == std::vector<int>({2, 3}));
    assert(are_tensors_equal(add_result1, Tensor::create(std::vector<int>{2, 3}, std::make_shared<std::vector<float>>(std::vector<float>{3.0, 4.0, 5.0, 6.0, 7.0, 8.0}))));
    std::cout << "Addition broadcasting 1x1 to 2x3 test passed." << std::endl;

    auto vector_1d = Tensor::create(std::vector<int>{1, 3}, std::make_shared<std::vector<float>>(std::vector<float>{1.0, 2.0, 3.0}));
    auto add_result2 = (*vector_1d) + matrix_2d;
    assert(add_result2->get_shape() == std::vector<int>({2, 3}));
    assert(are_tensors_equal(add_result2, Tensor::create(std::vector<int>{2, 3}, std::make_shared<std::vector<float>>(std::vector<float>{2.0, 4.0, 6.0, 5.0, 7.0, 9.0}))));
    std::cout << "Addition broadcasting 1x3 to 2x3 test passed." << std::endl;

    auto tensor_3d_a = Tensor::create(std::vector<int>{2, 1, 3}, std::make_shared<std::vector<float>>(std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
    auto tensor_3d_b = Tensor::create(std::vector<int>{2, 2, 1}, std::make_shared<std::vector<float>>(std::vector<float>{1.0, 2.0, 3.0, 4.0}));
    auto add_result3 = (*tensor_3d_a) + tensor_3d_b;
    assert(add_result3->get_shape() == std::vector<int>({2, 2, 3}));
    assert(are_tensors_equal(add_result3, Tensor::create(std::vector<int>{2, 2, 3}, std::make_shared<std::vector<float>>(std::vector<float>{2.0, 3.0, 4.0,
                                                                                                                                            3.0, 4.0, 5.0,
                                                                                                                                            7.0, 8.0, 9.0,
                                                                                                                                            8.0, 9.0, 10.0}))));
    std::cout << "Addition broadcasting 3D tensors test passed." << std::endl;
}

void test_subtraction()
{
    std::cout << "\nTest Case: Subtraction" << std::endl;
    auto scalar_like = Tensor::create(std::vector<int>{1, 1}, std::make_shared<std::vector<float>>(std::vector<float>{2.0}));
    auto matrix_2d = Tensor::create(std::vector<int>{2, 3}, std::make_shared<std::vector<float>>(std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
    auto sub_result1 = (*matrix_2d) - scalar_like;
    assert(sub_result1->get_shape() == std::vector<int>({2, 3}));
    assert(are_tensors_equal(sub_result1, Tensor::create(std::vector<int>{2, 3}, std::make_shared<std::vector<float>>(std::vector<float>{-1.0, 0.0, 1.0, 2.0, 3.0, 4.0}))));
    std::cout << "Subtraction broadcasting 1x1 from 2x3 test passed." << std::endl;

    auto vector_1d = Tensor::create(std::vector<int>{1, 3}, std::make_shared<std::vector<float>>(std::vector<float>{1.0, 2.0, 3.0}));
    auto sub_result2 = (*matrix_2d) - vector_1d;
    assert(sub_result2->get_shape() == std::vector<int>({2, 3}));
    assert(are_tensors_equal(sub_result2, Tensor::create(std::vector<int>{2, 3}, std::make_shared<std::vector<float>>(std::vector<float>{0.0, 0.0, 0.0, 3.0, 3.0, 3.0}))));
    std::cout << "Subtraction broadcasting 1x3 from 2x3 test passed." << std::endl;

    auto tensor_3d_a = Tensor::create(std::vector<int>{2, 1, 3}, std::make_shared<std::vector<float>>(std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
    auto tensor_3d_b = Tensor::create(std::vector<int>{2, 2, 1}, std::make_shared<std::vector<float>>(std::vector<float>{1.0, 2.0, 3.0, 4.0}));
    auto sub_result3 = (*tensor_3d_a) - tensor_3d_b;
    assert(sub_result3->get_shape() == std::vector<int>({2, 2, 3}));
    assert(are_tensors_equal(sub_result3, Tensor::create(std::vector<int>{2, 2, 3}, std::make_shared<std::vector<float>>(std::vector<float>{0.0, 1.0, 2.0,
                                                                                                                                            -1.0, 0.0, 1.0,
                                                                                                                                            1.0, 2.0, 3.0,
                                                                                                                                            0.0, 1.0, 2.0}))));
    std::cout << "Subtraction broadcasting 3D tensors test passed." << std::endl;
}

void test_dot_product()
{
    std::cout << "\nTest Case: Dot Product (Matrix Multiplication)" << std::endl;
    auto mat_a = Tensor::create(std::vector<int>{2, 3}, std::make_shared<std::vector<float>>(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    auto mat_b = Tensor::create(std::vector<int>{3, 2}, std::make_shared<std::vector<float>>(std::vector<float>{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}));

    auto dot_result_2d = mat_a->dot(mat_b);
    assert(dot_result_2d->get_shape() == std::vector<int>({2, 2}));
    assert(are_tensors_equal(dot_result_2d, Tensor::create(std::vector<int>{2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{58.0f, 64.0f, 139.0f, 154.0f}))));
    std::cout << "2D Matrix multiplication test passed." << std::endl;

    auto batch_mat_a = Tensor::create(std::vector<int>{2, 2, 3}, std::make_shared<std::vector<float>>(std::vector<float>{1.0f, 2.0f, 3.0f,
                                                                                                                         4.0f, 5.0f, 6.0f,
                                                                                                                         -1.0f, -2.0f, -3.0f,
                                                                                                                         -4.0f, -5.0f, -6.0f}));
    auto batch_mat_b = Tensor::create(std::vector<int>{2, 3, 2}, std::make_shared<std::vector<float>>(std::vector<float>{7.0f, 8.0f,
                                                                                                                         9.0f, 10.0f,
                                                                                                                         11.0f, 12.0f,
                                                                                                                         -7.0f, -8.0f,
                                                                                                                         -9.0f, -10.0f,
                                                                                                                         -11.0f, -12.0f}));

    auto dot_result_3d = batch_mat_a->dot(batch_mat_b);
    assert(dot_result_3d->get_shape() == std::vector<int>({2, 2, 2}));
    assert(are_tensors_equal(dot_result_3d, Tensor::create(std::vector<int>{2, 2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{58.0f, 64.0f,
                                                                                                                                              139.0f, 154.0f,
                                                                                                                                              58.0f, 64.0f,
                                                                                                                                              139.0f, 154.0f}))));
    std::cout << "3D x 3D Batched matrix multiplication test passed." << std::endl;

    auto mat_a_broadcast = Tensor::create(std::vector<int>{2, 3}, std::make_shared<std::vector<float>>(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    auto batch_mat_b_broadcast = Tensor::create(std::vector<int>{2, 3, 2}, std::make_shared<std::vector<float>>(std::vector<float>{7.0f, 8.0f,
                                                                                                                                   9.0f, 10.0f,
                                                                                                                                   11.0f, 12.0f,
                                                                                                                                   -7.0f, -8.0f,
                                                                                                                                   -9.0f, -10.0f,
                                                                                                                                   -11.0f, -12.0f}));

    auto dot_result_broadcast = mat_a_broadcast->dot(batch_mat_b_broadcast);
    assert(dot_result_broadcast->get_shape() == std::vector<int>({2, 2, 2}));
    assert(are_tensors_equal(dot_result_broadcast, Tensor::create(std::vector<int>{2, 2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{58.0f, 64.0f,
                                                                                                                                                     139.0f, 154.0f,
                                                                                                                                                     -58.0f, -64.0f,
                                                                                                                                                     -139.0f, -154.0f}))));
    std::cout << "2D x 3D Batched matrix multiplication (broadcasting) test passed." << std::endl;

    auto A = Tensor::create(std::vector<int>{2, 2, 3}, std::make_shared<std::vector<float>>(std::vector<float>{1, 2, 3,
                                                                                                               4, 5, 6,
                                                                                                               -1, -2, -3,
                                                                                                               -4, -5, -6}));

    auto B = Tensor::create(std::vector<int>{3, 2}, std::make_shared<std::vector<float>>(std::vector<float>{
                                                        7,
                                                        8,
                                                        9,
                                                        10,
                                                        11,
                                                        12,
                                                    }));
    auto C = A->dot(B);

    assert(C->get_shape() == std::vector<int>({2, 2, 2}));
    assert(are_tensors_equal(C, Tensor::create(std::vector<int>{2, 2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{58, 64,
                                                                                                                                  139, 154,
                                                                                                                                  -58, -64,
                                                                                                                                  -139, -154}))));
    std::cout << "3D x 2D Batched matrix multiplication (broadcasting) test passed." << std::endl;
}

void test_transpose()
{
    std::cout << "\nTest Case: Transpose" << std::endl;
    auto trans_t1 = Tensor::create(std::vector<int>{2, 3}, std::make_shared<std::vector<float>>(std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
    auto trans_result1 = trans_t1->transpose({1, 0});
    assert(trans_result1->get_shape() == std::vector<int>({3, 2}));
    assert(are_tensors_equal(trans_result1, Tensor::create(std::vector<int>{3, 2}, std::make_shared<std::vector<float>>(std::vector<float>{1.0, 4.0, 2.0, 5.0, 3.0, 6.0}))));
    std::cout << "2D Transpose test passed." << std::endl;

    auto trans_t2 = Tensor::create(std::vector<int>{2, 3, 4});
    auto trans_result2 = trans_t2->transpose({2, 1, 0});
    assert(trans_result2->get_shape() == std::vector<int>({4, 3, 2}));
    std::cout << "3D Transpose test passed." << std::endl;
}

void test_reshape()
{
    std::cout << "\nTest Case: Reshape" << std::endl;
    auto reshape_t1 = Tensor::create(std::vector<int>{2, 3}, std::make_shared<std::vector<float>>(std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
    auto reshape_result1 = reshape_t1->reshape({6, 1});
    assert(reshape_result1->get_shape() == std::vector<int>({6, 1}));
    assert(are_tensors_equal(reshape_result1, Tensor::create(std::vector<int>{6, 1}, std::make_shared<std::vector<float>>(std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}))));
    std::cout << "Reshape 2x3 to 6x1 test passed." << std::endl;

    auto reshape_result2 = reshape_t1->reshape({3, 2});
    assert(reshape_result2->get_shape() == std::vector<int>({3, 2}));
    assert(are_tensors_equal(reshape_result2, Tensor::create(std::vector<int>{3, 2}, std::make_shared<std::vector<float>>(std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}))));
    std::cout << "Reshape 2x3 to 3x2 test passed." << std::endl;

    auto reshape_t2 = Tensor::create(std::vector<int>{2, 2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}));
    auto reshape_result3 = reshape_t2->reshape({4, 2});
    assert(reshape_result3->get_shape() == std::vector<int>({4, 2}));
    assert(are_tensors_equal(reshape_result3, Tensor::create(std::vector<int>{4, 2}, std::make_shared<std::vector<float>>(std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}))));
    std::cout << "Reshape 2x2x2 to 4x2 test passed." << std::endl;
}

void test_sum()
{
    std::cout << "\nTest Case: Sum" << std::endl;

    auto t1d = Tensor::create(std::vector<int>{5}, std::make_shared<std::vector<float>>(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}));
    auto sum_result1 = t1d->sum();

    assert(sum_result1->get_shape() == std::vector<int>({1}));
    assert(are_tensors_equal(sum_result1, Tensor::create(std::vector<int>{1}, std::make_shared<std::vector<float>>(std::vector<float>{15.0f}))));
    std::cout << "  1D Tensor sum forward pass passed." << std::endl;

    auto t2d = Tensor::create(std::vector<int>{2, 3}, std::make_shared<std::vector<float>>(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    auto sum_result2 = t2d->sum();

    assert(sum_result2->get_shape() == std::vector<int>({1}));
    assert(are_tensors_equal(sum_result2, Tensor::create(std::vector<int>{1}, std::make_shared<std::vector<float>>(std::vector<float>{21.0f}))));
    std::cout << "  2D Tensor sum forward pass passed." << std::endl;

    auto t_empty = Tensor::create(std::vector<int>{0}, std::make_shared<std::vector<float>>(std::vector<float>{}));
    auto sum_result_empty = t_empty->sum();
    assert(sum_result_empty->get_shape() == std::vector<int>({1}));
    assert(are_tensors_equal(sum_result_empty, Tensor::create(std::vector<int>{1}, std::make_shared<std::vector<float>>(std::vector<float>{0.0f}))));
    std::cout << "  Empty Tensor sum forward pass passed." << std::endl;

    std::cout << "Sum test completed." << std::endl;
}

void test_division()
{
    std::cout << "\nTest Case: Division" << std::endl;

    auto A = Tensor::create(std::vector<int>{2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{4.0f, 6.0f, 8.0f, 10.0f}));
    auto B = Tensor::create(std::vector<int>{2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{2.0f, 3.0f, 4.0f, 5.0f}));

    auto C = (*A) / B;

    assert(C->get_shape() == std::vector<int>({2, 2}));
    assert(are_tensors_equal(C, Tensor::create(std::vector<int>{2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{2.0f, 2.0f, 2.0f, 2.0f}))));
    std::cout << "  Element-wise division forward pass passed." << std::endl;

    auto D = Tensor::create(std::vector<int>{2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f}));
    auto E = Tensor::create(std::vector<int>{1}, std::make_shared<std::vector<float>>(std::vector<float>{10.0f}));

    auto F = (*D) / E;

    assert(F->get_shape() == std::vector<int>({2, 2}));
    assert(are_tensors_equal(F, Tensor::create(std::vector<int>{2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}))));
    std::cout << "  Division with scalar broadcasting forward pass passed." << std::endl;

    auto G = Tensor::create(std::vector<int>{2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f}));
    auto H = Tensor::create(std::vector<int>{1, 2}, std::make_shared<std::vector<float>>(std::vector<float>{2.0f, 5.0f}));

    auto I = (*G) / H;

    assert(I->get_shape() == std::vector<int>({2, 2}));
    assert(are_tensors_equal(I, Tensor::create(std::vector<int>{2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{5.0f, 4.0f, 15.0f, 8.0f}))));
    std::cout << "  Division with row vector broadcasting forward pass passed." << std::endl;

    std::cout << "Division test completed." << std::endl;
}

void test_backward_pass()
{
    std::cout << "\nTest Case: Backward Pass through a Chain of Operations" << std::endl;

    auto A = Tensor::create(std::vector<int>{2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}), true);
    auto B = Tensor::create(std::vector<int>{2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{0.5f, 1.0f, 1.5f, 2.0f}), true);
    auto C = Tensor::create(std::vector<int>{2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{2.0f, 1.0f, 1.0f, 2.0f}), true);
    auto D = Tensor::create(std::vector<int>{2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f}), true);
    auto E = Tensor::create(std::vector<int>{1}, std::make_shared<std::vector<float>>(std::vector<float>{10.0f}), true);

    auto T1 = (*A) + B;
    auto T2 = (*T1) * C;
    auto T3 = T2->dot(D);
    auto T4 = T3->transpose({1, 0});
    auto T5 = T4->reshape({4, 1});
    auto T6 = (*T5) / E;
    auto final_loss = T6->sum();

    assert(final_loss->get_shape() == std::vector<int>({1}));
    assert(are_tensors_equal(final_loss, Tensor::create(std::vector<int>{1}, std::make_shared<std::vector<float>>(std::vector<float>{2.25f}))));
    std::cout << "  Complex chain forward pass passed." << std::endl;

    auto grad_Final_Loss = Tensor::create(std::vector<int>{1}, std::make_shared<std::vector<float>>(std::vector<float>{1.0f}));
    final_loss->backward(grad_Final_Loss);

    auto expected_grad_A = Tensor::create(std::vector<int>{2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{0.2f, 0.1f, 0.1f, 0.2f}));
    auto expected_grad_B = Tensor::create(std::vector<int>{2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{0.2f, 0.1f, 0.1f, 0.2f}));
    auto expected_grad_C = Tensor::create(std::vector<int>{2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{0.15f, 0.3f, 0.45f, 0.6f}));
    auto expected_grad_D = Tensor::create(std::vector<int>{2, 2}, std::make_shared<std::vector<float>>(std::vector<float>{0.75f, 0.75f, 1.5f, 1.5f}));
    auto expected_grad_E = Tensor::create(std::vector<int>{1}, std::make_shared<std::vector<float>>(std::vector<float>{-0.225f}));

    assert(are_tensors_equal(Tensor::create(A->get_shape(), std::make_shared<std::vector<float>>(A->get_grad())), expected_grad_A));
    std::cout << "  Gradient of A verified." << std::endl;
    assert(are_tensors_equal(Tensor::create(B->get_shape(), std::make_shared<std::vector<float>>(B->get_grad())), expected_grad_B));
    std::cout << "  Gradient of B verified." << std::endl;
    assert(are_tensors_equal(Tensor::create(C->get_shape(), std::make_shared<std::vector<float>>(C->get_grad())), expected_grad_C, 1e-3));
    std::cout << "  Gradient of C verified." << std::endl;
    assert(are_tensors_equal(Tensor::create(D->get_shape(), std::make_shared<std::vector<float>>(D->get_grad())), expected_grad_D));
    std::cout << "  Gradient of D verified." << std::endl;
    assert(are_tensors_equal(Tensor::create(E->get_shape(), std::make_shared<std::vector<float>>(E->get_grad())), expected_grad_E));
    std::cout << "  Gradient of E verified." << std::endl;

    std::cout << "Backward Pass through a Chain of Operations test passed." << std::endl;
}

void test_backward_pass_with_broadcasting_operations()
{
    std::cout << "\nTest Case: Backward Pass with Broadcasting" << std::endl;

    auto A = Tensor::create(std::vector<int>{2, 3}, std::make_shared<std::vector<float>>(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}), true);
    auto B = Tensor::create(std::vector<int>{3}, std::make_shared<std::vector<float>>(std::vector<float>{0.5f, 1.0f, 1.5f}), true);
    auto C = Tensor::create(std::vector<int>{2, 1}, std::make_shared<std::vector<float>>(std::vector<float>{2.0f, 3.0f}), true);
    auto D = Tensor::create(std::vector<int>{1, 3}, std::make_shared<std::vector<float>>(std::vector<float>{1.0f, 2.0f, 3.0f}), true);
    auto E = Tensor::create(std::vector<int>{1}, std::make_shared<std::vector<float>>(std::vector<float>{10.0f}), true);

    auto T1 = (*A) + B;
    auto T2 = (*T1) - D;
    auto T3 = (*T2) * C;
    auto T4 = (*T3) / E;
    auto T5 = T4->sum();
    auto final = T5->reshape({1, 1});

    assert(final->get_shape() == std::vector<int>({1, 1}));
    assert(are_tensors_equal(final, Tensor::create(std::vector<int>{1, 1}, std::make_shared<std::vector<float>>(std::vector<float>{4.2f}))));
    std::cout << "  Complex chain with broadcasting forward pass passed." << std::endl;

    auto grad_final = Tensor::create(std::vector<int>{1, 1}, std::make_shared<std::vector<float>>(std::vector<float>{1.0f}));
    final->backward(grad_final);

    auto expected_grad_A = Tensor::create(std::vector<int>{2, 3}, std::make_shared<std::vector<float>>(std::vector<float>{0.2f, 0.2f, 0.2f, 0.3f, 0.3f, 0.3f}));
    auto expected_grad_B = Tensor::create(std::vector<int>{3}, std::make_shared<std::vector<float>>(std::vector<float>{0.5f, 0.5f, 0.5f}));
    auto expected_grad_C = Tensor::create(std::vector<int>{2, 1}, std::make_shared<std::vector<float>>(std::vector<float>{0.3f, 1.2f}));
    auto expected_grad_D = Tensor::create(std::vector<int>{1, 3}, std::make_shared<std::vector<float>>(std::vector<float>{-0.5f, -0.5f, -0.5f}));
    auto expected_grad_E = Tensor::create(std::vector<int>{1}, std::make_shared<std::vector<float>>(std::vector<float>{-0.42f}));

    assert(are_tensors_equal(Tensor::create(A->get_shape(), std::make_shared<std::vector<float>>(A->get_grad())), expected_grad_A));
    std::cout << "  Gradient of A verified." << std::endl;
    assert(are_tensors_equal(Tensor::create(B->get_shape(), std::make_shared<std::vector<float>>(B->get_grad())), expected_grad_B, 1e-6));
    std::cout << "  Gradient of B verified." << std::endl;
    assert(are_tensors_equal(Tensor::create(C->get_shape(), std::make_shared<std::vector<float>>(C->get_grad())), expected_grad_C, 1e-6));
    std::cout << "  Gradient of D verified." << std::endl;
    assert(are_tensors_equal(Tensor::create(D->get_shape(), std::make_shared<std::vector<float>>(D->get_grad())), expected_grad_D, 1e-6));
    std::cout << "  Gradient of E verified." << std::endl;
    assert(are_tensors_equal(Tensor::create(E->get_shape(), std::make_shared<std::vector<float>>(E->get_grad())), expected_grad_E, 1e-6));
    std::cout << "Backward Pass with Broadcasting over test passed." << std::endl;
}

int main()
{
    std::cout << "Starting Tensor class tests..." << std::endl;

    test_addition();
    test_subtraction();
    test_dot_product();
    test_transpose();
    test_reshape();
    test_sum();
    test_division();
    test_backward_pass();
    test_backward_pass_with_broadcasting_operations();

    std::cout << "\nAll Tensor class tests completed." << std::endl;
    return 0;
}
