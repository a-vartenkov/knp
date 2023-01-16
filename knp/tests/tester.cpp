/**
 * Tests runner.
 **/

#include <gtest/gtest.h>

#include <exception>
#include <iostream>


int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    try
    {
        return RUN_ALL_TESTS();
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << e.what() << std::endl;
        throw;
    }

    return EXIT_SUCCESS;
}