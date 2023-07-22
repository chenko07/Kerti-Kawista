<?php

include '../connect.php';
session_start();
if(isset($_SESSION['status']) != "login"){
    header("location: ../index.php");
}

echo "Selamat datang ". $_SESSION['username'];
?>

<a href="logout.php">LOGOUT</a>