<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login User</title>
</head>
<body>
    <h2>LOGIN USER</h2>
    <form action="encrypt.php" method="post">
        <div>
            <label for="username">Username</label>
            <input type="text" name="username" id="username" placeholder="Masukkan Username">
        </div>
        <div>
            <label for="password">Password</label>
            <input type="password" name="password" id="password" placeholder="Masukkan Password">
        </div>
        <div>
            <input type="submit" value="Login">
        </div>
    </form>
</body>
</html>