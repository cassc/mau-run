pragma solidity ^0.4.26;

/**
 * @title OverflowUnderflowDemo
 * @dev This contract demonstrates several different
 * integer overflow/underflow scenarios. Compile using
 * a Solidity compiler version below 0.8.x to witness
 * the buggy wrap-around.
 */
contract OverflowUnderflowDemo {

    // We keep track of the "result" in storage so that
    // it's easily readable after each test call.
    uint256 public result;
    // Signed integer variant to demonstrate signed overflow
    int256 public signedResult;

    /**
     * @dev Demonstrates a simple overflow in unsigned addition.
     * For example, if you call addOverflow(2**256 - 1, 10),
     * it will wrap around to a small number instead of reverting.
     */
    function addOverflow(uint256 _a, uint256 _b) public returns (uint256) {
        result = _a + _b; // Overflows silently in <0.8.x
        return result;
    }

    /**
     * @dev Demonstrates an underflow with unsigned subtraction.
     * For instance, subUnderflow(5, 10) will wrap around to
     * a huge number near 2**256 - 1 in <0.8.x.
     */
    function subUnderflow(uint256 _a, uint256 _b) public returns (uint256) {
        result = _a - _b; // Underflows silently in <0.8.x
        return result;
    }

    /**
     * @dev Demonstrates a multiplication overflow in unsigned integers.
     * For example, mulOverflow(2**128, 2**128) will exceed the max
     * uint256, and wrap around in <0.8.x.
     */
    function mulOverflow(uint256 _a, uint256 _b) public returns (uint256) {
        result = _a * _b; // Overflows silently in <0.8.x
        return result;
    }

    /**
     * @dev Demonstrates exponent-based overflow, which is just
     * a specialized case of multiplication repeated many times.
     * e.g., expOverflow(2, 256) in <0.8.x might wrap around to 0.
     */
    function expOverflow(uint256 base, uint256 exponent) public returns (uint256) {
        // Naive exponentiation may overflow.
        // This function calculates base**exponent in a loop
        // just for demonstration (and is obviously not gas-efficient).
        uint256 r = 1;
        for(uint256 i = 0; i < exponent; i++) {
            r = r * base; // possible repeated overflow
        }
        result = r;
        return result;
    }

    /**
     * @dev Demonstrates signed integer overflow, which can happen
     * if you pass large positive or negative values near the limits of int256.
     * e.g., signedAddOverflow(2**255 - 1, 100) in <0.8.x would wrap to negative.
     */
    function signedAddOverflow(int256 _a, int256 _b) public returns (int256) {
        signedResult = _a + _b; // Overflow in <0.8.x
        return signedResult;
    }

    /**
     * @dev Demonstrates signed integer underflow.
     * e.g., signedSubUnderflow(-2**255, 1) in <0.8.x would wrap to a large positive number.
     */
    function signedSubUnderflow(int256 _a, int256 _b) public returns (int256) {
        signedResult = _a - _b; // Underflow in <0.8.x
        return signedResult;
    }
}
